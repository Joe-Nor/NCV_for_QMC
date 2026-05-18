!-----------------------------------------------------------------!
! Parity prefix + candidate-wise ΔK computation library           !
! For use from Python via ctypes                                   !
!                                                                  !
! Extension of parity_prefix_lib.f90:                              !
!   In addition to the standard parity_prefix and deltaK_prefix,   !
!   computes deltaK for ALL candidate bonds at EACH autoregressive !
!   position. Uses save/restore on the vertex graph to avoid       !
!   rebuilding — O(nh * nb * L_avg) total.                        !
!                                                                  !
! Independent module — does NOT depend on parity_prefix_mod.       !
!-----------------------------------------------------------------!

module parity_prefix_candidates_mod
  implicit none
  private

  ! Internal work arrays (module-level, independent from parity_prefix_mod)
  integer :: mm_w = 0, nn_w = 0, nb_w = 0
  integer :: mm_alloc = 0, nn_alloc = 0, nb_alloc = 0
  integer, allocatable :: opstring_w(:)
  integer, allocatable :: bsites_w(:,:)
  integer, allocatable :: frstspinop_w(:)
  integer, allocatable :: lastspinop_w(:)
  integer, allocatable :: vertexlist_w(:)

  ! Site-time linked lists
  integer, allocatable :: st_prev_w(:,:)
  integer, allocatable :: st_next_w(:,:)
  logical, allocatable :: st_used_w(:,:)
  integer, allocatable :: st_head_w(:)
  integer, allocatable :: st_tail_w(:)
  integer, allocatable :: st_cursor_w(:)

  public :: compute_parity_prefix_candidates_impl

contains

!------------------------------------------!
! Memory management
!------------------------------------------!

subroutine alloc_work()
  logical :: need_mm, need_nn

  need_mm = (mm_w > mm_alloc)
  need_nn = (nn_w > nn_alloc) .or. (nb_w > nb_alloc)

  if (need_mm) then
     if (allocated(opstring_w))   deallocate(opstring_w)
     if (allocated(vertexlist_w)) deallocate(vertexlist_w)
     if (allocated(st_prev_w))    deallocate(st_prev_w)
     if (allocated(st_next_w))    deallocate(st_next_w)
     if (allocated(st_used_w))    deallocate(st_used_w)

     allocate(opstring_w(0:mm_w-1))
     allocate(vertexlist_w(0:4*mm_w-1))
     allocate(st_prev_w(2, 0:mm_w-1))
     allocate(st_next_w(2, 0:mm_w-1))
     allocate(st_used_w(2, 0:mm_w-1))
     mm_alloc = mm_w
  endif

  if (need_nn) then
     if (allocated(bsites_w))     deallocate(bsites_w)
     if (allocated(frstspinop_w)) deallocate(frstspinop_w)
     if (allocated(lastspinop_w)) deallocate(lastspinop_w)
     if (allocated(st_head_w))    deallocate(st_head_w)
     if (allocated(st_tail_w))    deallocate(st_tail_w)
     if (allocated(st_cursor_w))  deallocate(st_cursor_w)

     allocate(bsites_w(2, nb_w))
     allocate(frstspinop_w(nn_w))
     allocate(lastspinop_w(nn_w))
     allocate(st_head_w(nn_w))
     allocate(st_tail_w(nn_w))
     allocate(st_cursor_w(nn_w))
     nn_alloc = nn_w
     nb_alloc = nb_w
  endif
end subroutine alloc_work

subroutine init_empty_vertex_graph()
  vertexlist_w(0:4*mm_w-1) = -1
  frstspinop_w(1:nn_w) = -1
  lastspinop_w(1:nn_w) = -1
  st_prev_w(:, 0:mm_w-1) = -1
  st_next_w(:, 0:mm_w-1) = -1
  st_used_w(:, 0:mm_w-1) = .false.
  st_head_w(1:nn_w) = -1
  st_tail_w(1:nn_w) = -1
  st_cursor_w(1:nn_w) = -1
end subroutine init_empty_vertex_graph

!------------------------------------------!
! Site-time linked list operations
!------------------------------------------!

integer function st_slot_of_site_occupied_w(site, p)
  integer, intent(in) :: site, p
  integer :: b
  b = opstring_w(p) / 2
  if (bsites_w(1,b) == site) then
     st_slot_of_site_occupied_w = 1
  else
     st_slot_of_site_occupied_w = 2
  endif
end function st_slot_of_site_occupied_w

integer function st_slot_of_site_bond_w(site, bond_b)
  integer, intent(in) :: site, bond_b
  if (bsites_w(1,bond_b) == site) then
     st_slot_of_site_bond_w = 1
  else
     st_slot_of_site_bond_w = 2
  endif
end function st_slot_of_site_bond_w

integer function st_prev_pos_w(site, p)
  integer, intent(in) :: site, p
  integer :: slot
  slot = st_slot_of_site_occupied_w(site, p)
  st_prev_pos_w = st_prev_w(slot, p)
end function st_prev_pos_w

integer function st_next_pos_w(site, p)
  integer, intent(in) :: site, p
  integer :: slot
  slot = st_slot_of_site_occupied_w(site, p)
  st_next_pos_w = st_next_w(slot, p)
end function st_next_pos_w

subroutine st_ensure_cursor_w(site, q)
  integer, intent(in) :: site, q
  integer :: c, n

  if (st_head_w(site) == -1) then
     st_cursor_w(site) = -1
     return
  endif

  c = st_cursor_w(site)
  if (c == -1) c = st_head_w(site)

  do while (c < q)
     if (c == st_tail_w(site)) then
        c = st_head_w(site)
        exit
     else
        n = st_next_pos_w(site, c)
        c = n
     endif
  enddo

  st_cursor_w(site) = c
end subroutine st_ensure_cursor_w

subroutine st_insert_site_w(site, p, slot, prevp, nextp)
  integer, intent(in) :: site, p, slot, prevp, nextp
  integer :: prev_slot, next_slot

  if (st_head_w(site) == -1) then
     st_head_w(site) = p
     st_tail_w(site) = p
     st_cursor_w(site) = p
     st_prev_w(slot,p) = p
     st_next_w(slot,p) = p
     st_used_w(slot,p) = .true.
     return
  endif

  prev_slot = st_slot_of_site_occupied_w(site, prevp)
  next_slot = st_slot_of_site_occupied_w(site, nextp)

  st_prev_w(slot,p) = prevp
  st_next_w(slot,p) = nextp
  st_used_w(slot,p) = .true.

  st_next_w(prev_slot, prevp) = p
  st_prev_w(next_slot, nextp) = p

  if (p < st_head_w(site)) st_head_w(site) = p
  if (p > st_tail_w(site)) st_tail_w(site) = p
  st_cursor_w(site) = p
end subroutine st_insert_site_w

!------------------------------------------!
! Vertex graph operations
!------------------------------------------!

integer function leg_in_for_site_w(site, p)
  integer, intent(in) :: site, p
  integer :: b
  b = opstring_w(p) / 2
  if (bsites_w(1,b) == site) then
     leg_in_for_site_w = 4*p
  else
     leg_in_for_site_w = 4*p + 1
  endif
end function leg_in_for_site_w

integer function leg_out_for_site_w(site, p)
  integer, intent(in) :: site, p
  integer :: b
  b = opstring_w(p) / 2
  if (bsites_w(1,b) == site) then
     leg_out_for_site_w = 4*p + 2
  else
     leg_out_for_site_w = 4*p + 3
  endif
end function leg_out_for_site_w

subroutine insert_one_site_w(site, vin, vout, prev, next)
  integer, intent(in) :: site, vin, vout, prev, next
  integer :: prev_eff, next_eff
  if (frstspinop_w(site) == -1) then
     frstspinop_w(site) = vin; lastspinop_w(site) = vout
     vertexlist_w(vin) = vout; vertexlist_w(vout) = vin
  else
     prev_eff = prev; if (prev_eff == -1) prev_eff = lastspinop_w(site)
     next_eff = next; if (next_eff == -1) next_eff = frstspinop_w(site)
     vertexlist_w(prev_eff) = vin; vertexlist_w(vin) = prev_eff
     vertexlist_w(vout) = next_eff; vertexlist_w(next_eff) = vout
     if (prev == -1) frstspinop_w(site) = vin
     if (next == -1) lastspinop_w(site) = vout
  endif
end subroutine insert_one_site_w

!------------------------------------------!
! Core update routines (same as original)
!------------------------------------------!

subroutine do_update_insert(pos, bond_b)
  integer, intent(in) :: pos, bond_b
  integer :: s1, s2, v0
  integer :: c1, c2, prevp1, nextp1, prevp2, nextp2
  integer :: prev1, next1, prev2, next2

  v0 = 4*pos
  s1 = bsites_w(1, bond_b)
  s2 = bsites_w(2, bond_b)

  call st_ensure_cursor_w(s1, pos)
  if (st_head_w(s1) == -1) then
     prevp1 = -1; nextp1 = -1; prev1 = -1; next1 = -1
  else
     c1 = st_cursor_w(s1)
     nextp1 = c1
     prevp1 = st_prev_pos_w(s1, c1)
     prev1  = leg_out_for_site_w(s1, prevp1)
     next1  = leg_in_for_site_w(s1, nextp1)
  endif

  call st_ensure_cursor_w(s2, pos)
  if (st_head_w(s2) == -1) then
     prevp2 = -1; nextp2 = -1; prev2 = -1; next2 = -1
  else
     c2 = st_cursor_w(s2)
     nextp2 = c2
     prevp2 = st_prev_pos_w(s2, c2)
     prev2  = leg_out_for_site_w(s2, prevp2)
     next2  = leg_in_for_site_w(s2, nextp2)
  endif

  call insert_one_site_w(s1, v0, v0+2, prev1, next1)
  call insert_one_site_w(s2, v0+1, v0+3, prev2, next2)

  if (prevp1 == -1) then
     call st_insert_site_w(s1, pos, 1, -1, -1)
  else
     call st_insert_site_w(s1, pos, 1, prevp1, nextp1)
  endif

  if (prevp2 == -1) then
     call st_insert_site_w(s2, pos, 2, -1, -1)
  else
     call st_insert_site_w(s2, pos, 2, prevp2, nextp2)
  endif
end subroutine do_update_insert

subroutine do_delta_loops_insert(pos, bond_b, delta_k)
  integer, intent(in) :: pos, bond_b
  integer, intent(out) :: delta_k
  integer :: s1, s2, v0, prev1, next1, prev2, next2, v1, v2, first_hit

  v0 = 4*pos
  s1 = bsites_w(1, bond_b)
  s2 = bsites_w(2, bond_b)
  prev1 = vertexlist_w(v0)
  next1 = vertexlist_w(v0+2)
  prev2 = vertexlist_w(v0+1)
  next2 = vertexlist_w(v0+3)

  ! Special case: site was free spin (self-loop)
  if (prev1 == v0+2 .or. prev2 == v0+3) then
     delta_k = -1
     return
  endif

  ! General case: traverse loop from v0 to find which target we hit first
  v1 = v0
  do
     v2 = vertexlist_w(v1)
     v1 = ieor(v2, 1)
     if (v1 == prev2 .or. v1 == next1 .or. v1 == next2) then
        first_hit = v1
        exit
     endif
  enddo

  if (first_hit == next1) then
     delta_k = -1
  else if (first_hit == prev2) then
     delta_k = +1
  else if (first_hit == next2) then
     delta_k = 0
  endif
end subroutine do_delta_loops_insert

!------------------------------------------!
! Candidate-only insert (skip cursor scan)
!------------------------------------------!

subroutine do_update_insert_at_end(pos, bond_b)
  !
  ! Same as do_update_insert, but assumes pos > all existing entries
  ! in the site-time linked lists. Skips cursor scan entirely — O(1).
  !
  integer, intent(in) :: pos, bond_b
  integer :: s1, s2, v0
  integer :: prevp1, nextp1, prevp2, nextp2
  integer :: prev1, next1, prev2, next2

  v0 = 4*pos
  s1 = bsites_w(1, bond_b)
  s2 = bsites_w(2, bond_b)

  ! Since pos > all existing entries, insertion wraps: prev=tail, next=head
  if (st_head_w(s1) == -1) then
     prevp1 = -1; nextp1 = -1; prev1 = -1; next1 = -1
  else
     nextp1 = st_head_w(s1)
     prevp1 = st_tail_w(s1)
     prev1 = leg_out_for_site_w(s1, prevp1)
     next1 = leg_in_for_site_w(s1, nextp1)
  endif

  if (st_head_w(s2) == -1) then
     prevp2 = -1; nextp2 = -1; prev2 = -1; next2 = -1
  else
     nextp2 = st_head_w(s2)
     prevp2 = st_tail_w(s2)
     prev2 = leg_out_for_site_w(s2, prevp2)
     next2 = leg_in_for_site_w(s2, nextp2)
  endif

  call insert_one_site_w(s1, v0, v0+2, prev1, next1)
  call insert_one_site_w(s2, v0+1, v0+3, prev2, next2)

  if (prevp1 == -1) then
     call st_insert_site_w(s1, pos, 1, -1, -1)
  else
     call st_insert_site_w(s1, pos, 1, prevp1, nextp1)
  endif

  if (prevp2 == -1) then
     call st_insert_site_w(s2, pos, 2, -1, -1)
  else
     call st_insert_site_w(s2, pos, 2, prevp2, nextp2)
  endif
end subroutine do_update_insert_at_end

!------------------------------------------!
! Candidate ΔK with save/restore
!------------------------------------------!

subroutine eval_candidate_deltaK(pos, bond_b, delta_k)
  !
  ! Compute ΔK for a hypothetical insertion of bond_b at position pos,
  ! WITHOUT permanently modifying the vertex graph.
  !
  ! Method: save affected state → insert → compute ΔK → restore.
  ! Cost: O(L_traverse) per call (the loop traversal in do_delta_loops_insert).
  !
  ! Precondition: pos > all currently occupied positions (append-only).
  ! Precondition: bsites_w(1,bond_b) /= bsites_w(2,bond_b) (no self-loops).
  !
  integer, intent(in) :: pos, bond_b
  integer, intent(out) :: delta_k

  integer :: s1, s2, v0
  integer :: nextp1, prevp1, nextp2, prevp2
  integer :: prev1, next1, prev2, next2
  integer :: prev1_slot, next1_slot, prev2_slot, next2_slot
  logical :: s1_free, s2_free

  ! Saved state
  integer :: sv_vl_prev1, sv_vl_next1, sv_vl_prev2, sv_vl_next2
  integer :: sv_st_next_prevp1, sv_st_prev_nextp1
  integer :: sv_st_next_prevp2, sv_st_prev_nextp2
  integer :: sv_head1, sv_tail1, sv_cursor1
  integer :: sv_head2, sv_tail2, sv_cursor2
  integer :: sv_frstspinop1, sv_lastspinop1
  integer :: sv_frstspinop2, sv_lastspinop2

  s1 = bsites_w(1, bond_b)
  s2 = bsites_w(2, bond_b)
  v0 = 4 * pos
  s1_free = (st_head_w(s1) == -1)
  s2_free = (st_head_w(s2) == -1)

  ! -------------------------------------------------------
  ! Save all state that may be modified by the insertion.
  ! Per-site: frstspinop, lastspinop, head, tail, cursor
  !           + vertexlist at prev/next legs (non-free sites only)
  !           + linked-list neighbor pointers (non-free sites only)
  ! -------------------------------------------------------
  sv_frstspinop1 = frstspinop_w(s1); sv_lastspinop1 = lastspinop_w(s1)
  sv_frstspinop2 = frstspinop_w(s2); sv_lastspinop2 = lastspinop_w(s2)
  sv_head1 = st_head_w(s1); sv_tail1 = st_tail_w(s1); sv_cursor1 = st_cursor_w(s1)
  sv_head2 = st_head_w(s2); sv_tail2 = st_tail_w(s2); sv_cursor2 = st_cursor_w(s2)

  if (.not. s1_free) then
     nextp1 = st_head_w(s1); prevp1 = st_tail_w(s1)
     prev1 = leg_out_for_site_w(s1, prevp1)
     next1 = leg_in_for_site_w(s1, nextp1)
     sv_vl_prev1 = vertexlist_w(prev1)
     sv_vl_next1 = vertexlist_w(next1)
     prev1_slot = st_slot_of_site_occupied_w(s1, prevp1)
     next1_slot = st_slot_of_site_occupied_w(s1, nextp1)
     sv_st_next_prevp1 = st_next_w(prev1_slot, prevp1)
     sv_st_prev_nextp1 = st_prev_w(next1_slot, nextp1)
  endif

  if (.not. s2_free) then
     nextp2 = st_head_w(s2); prevp2 = st_tail_w(s2)
     prev2 = leg_out_for_site_w(s2, prevp2)
     next2 = leg_in_for_site_w(s2, nextp2)
     sv_vl_prev2 = vertexlist_w(prev2)
     sv_vl_next2 = vertexlist_w(next2)
     prev2_slot = st_slot_of_site_occupied_w(s2, prevp2)
     next2_slot = st_slot_of_site_occupied_w(s2, nextp2)
     sv_st_next_prevp2 = st_next_w(prev2_slot, prevp2)
     sv_st_prev_nextp2 = st_prev_w(next2_slot, nextp2)
  endif

  ! -------------------------------------------------------
  ! Temporarily insert and compute ΔK
  ! -------------------------------------------------------
  call do_update_insert_at_end(pos, bond_b)
  opstring_w(pos) = 2 * bond_b
  call do_delta_loops_insert(pos, bond_b, delta_k)

  ! -------------------------------------------------------
  ! Restore all modified state
  ! -------------------------------------------------------
  frstspinop_w(s1) = sv_frstspinop1; lastspinop_w(s1) = sv_lastspinop1
  frstspinop_w(s2) = sv_frstspinop2; lastspinop_w(s2) = sv_lastspinop2
  st_head_w(s1) = sv_head1; st_tail_w(s1) = sv_tail1; st_cursor_w(s1) = sv_cursor1
  st_head_w(s2) = sv_head2; st_tail_w(s2) = sv_tail2; st_cursor_w(s2) = sv_cursor2

  ! Reset position-pos entries (were unoccupied before)
  vertexlist_w(v0)   = -1
  vertexlist_w(v0+1) = -1
  vertexlist_w(v0+2) = -1
  vertexlist_w(v0+3) = -1
  st_prev_w(:, pos)  = -1
  st_next_w(:, pos)  = -1
  st_used_w(:, pos)  = .false.
  opstring_w(pos)    = 0

  ! Restore vertexlist + linked-list neighbors for non-free sites
  if (.not. s1_free) then
     vertexlist_w(prev1) = sv_vl_prev1
     vertexlist_w(next1) = sv_vl_next1
     st_next_w(prev1_slot, prevp1) = sv_st_next_prevp1
     st_prev_w(next1_slot, nextp1) = sv_st_prev_nextp1
  endif

  if (.not. s2_free) then
     vertexlist_w(prev2) = sv_vl_prev2
     vertexlist_w(next2) = sv_vl_next2
     st_next_w(prev2_slot, prevp2) = sv_st_next_prevp2
     st_prev_w(next2_slot, nextp2) = sv_st_prev_nextp2
  endif

end subroutine eval_candidate_deltaK

!------------------------------------------!
! Main entry point
!------------------------------------------!

subroutine compute_parity_prefix_candidates_impl( &
    ops_compact, nh_in, bsites_in, nn_in, nb_in, &
    parity_prefix_out, deltaK_prefix_out, K_out, &
    deltaK_candidates_out)
  !
  ! Compute prefix parities, delta-K prefix, AND candidate-wise ΔK
  ! for all bonds at all autoregressive positions.
  !
  ! Inputs:
  !   ops_compact(nh_in)    - compact opstring (values = 2*b, 1-indexed bonds)
  !   bsites_in(2, nb_in)   - bond-to-site mapping (1-indexed)
  !   nn_in, nb_in          - number of sites, bonds
  !
  ! Outputs:
  !   parity_prefix_out(nh_in) - prefix parity at each step (int8: 0 or 1)
  !   deltaK_prefix_out(nh_in) - delta K at each step (-1, 0, +1)
  !   K_out                    - total loop count after all operators
  !   deltaK_candidates_out(nb_in, nh_in+1) - candidate ΔK for each bond
  !       Column j (1..nh_in+1) = autoregressive position j-1 (0..nh_in)
  !       Row i (1..nb_in) = candidate bond i
  !       Value = ΔK from inserting bond i after prefix of length (j-1)
  !
  ! Complexity: O(nh * L) for prefix + O(nh * nb * L_avg) for candidates
  !   where L is the loop traversal length. The factor nh from rebuilding
  !   the graph is eliminated by incremental save/restore.
  !
  integer, intent(in)    :: nh_in
  integer, intent(in)    :: ops_compact(nh_in)
  integer, intent(in)    :: nb_in
  integer, intent(in)    :: bsites_in(2, nb_in)
  integer, intent(in)    :: nn_in
  integer(1), intent(out) :: parity_prefix_out(nh_in)
  integer, intent(out)   :: deltaK_prefix_out(nh_in)
  integer, intent(out)   :: K_out
  integer, intent(out)   :: deltaK_candidates_out(nb_in, nh_in + 1)

  integer :: t, b, delta_k, prefix_parity, b_cand, dk_cand

  ! Handle empty opstring
  if (nh_in <= 0) then
     K_out = nn_in
     ! Position 0: all sites free → any bond gives ΔK = -1
     deltaK_candidates_out(:, 1) = -1
     return
  endif

  ! Set module-level parameters
  ! Need nh_in+1 positions: 0..nh_in-1 for actual operators,
  ! position nh_in for candidate evaluation at the last autoregressive step
  mm_w = nh_in + 1
  nn_w = nn_in
  nb_w = nb_in

  ! Allocate and initialize
  call alloc_work()
  bsites_w(:,:) = bsites_in(:,:)
  opstring_w(:) = 0
  call init_empty_vertex_graph()

  prefix_parity = 0
  K_out = nn_in  ! Start with nn free spins

  ! ----- Position 0 (empty graph): all sites free -----
  deltaK_candidates_out(:, 1) = -1

  ! ----- Incrementally insert operators -----
  do t = 1, nh_in
     b = ops_compact(t) / 2

     ! Insert actual operator at position t-1
     call do_update_insert(t-1, b)
     opstring_w(t-1) = ops_compact(t)
     call do_delta_loops_insert(t-1, b, delta_k)

     K_out = K_out + delta_k
     if (delta_k == 0) prefix_parity = 1 - prefix_parity

     parity_prefix_out(t) = int(prefix_parity, 1)
     deltaK_prefix_out(t) = delta_k

     ! ----- Candidate ΔK for position t (graph has t operators) -----
     ! Candidate insertion would be at opstring position t (one beyond last).
     ! eval_candidate_deltaK does save/restore, leaving graph unchanged.
     do b_cand = 1, nb_in
        call eval_candidate_deltaK(t, b_cand, dk_cand)
        deltaK_candidates_out(b_cand, t + 1) = dk_cand
     enddo
  enddo

end subroutine compute_parity_prefix_candidates_impl

end module parity_prefix_candidates_mod


!------------------------------------------!
! C-callable wrapper (for ctypes)
!------------------------------------------!
subroutine compute_parity_prefix_candidates( &
    ops_compact, nh_in, bsites_in, nn_in, nb_in, &
    parity_prefix_out, deltaK_prefix_out, K_out, &
    deltaK_candidates_out)
  use parity_prefix_candidates_mod, only: compute_parity_prefix_candidates_impl
  implicit none
  integer, intent(in)    :: nh_in
  integer, intent(in)    :: ops_compact(nh_in)
  integer, intent(in)    :: nb_in
  integer, intent(in)    :: bsites_in(2, nb_in)
  integer, intent(in)    :: nn_in
  integer(1), intent(out) :: parity_prefix_out(nh_in)
  integer, intent(out)   :: deltaK_prefix_out(nh_in)
  integer, intent(out)   :: K_out
  integer, intent(out)   :: deltaK_candidates_out(nb_in, nh_in + 1)

  call compute_parity_prefix_candidates_impl( &
       ops_compact, nh_in, bsites_in, nn_in, nb_in, &
       parity_prefix_out, deltaK_prefix_out, K_out, &
       deltaK_candidates_out)
end subroutine compute_parity_prefix_candidates
