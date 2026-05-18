!-----------------------------------------------------------------!
! Standalone parity_prefix computation library                     !
! For use from Python via ctypes                                   !
!                                                                  !
! Extracted from rsse_update_loops_cursor_optimized.f90             !
! Self-contained: no global state, all data passed via arguments   !
!-----------------------------------------------------------------!

module parity_prefix_mod
  implicit none
  private

  ! Internal work arrays (module-level, persist across calls)
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

  public :: compute_parity_prefix_impl

contains

!------------------------------------------!
subroutine compute_parity_prefix_impl( &
    ops_compact, nh_in, bsites_in, nn_in, nb_in, &
    parity_prefix_out, deltaK_prefix_out, K_out)
!------------------------------------------!
! Compute prefix parities and delta-K prefix for a compact opstring.
!
! Inputs:
!   ops_compact(nh_in)    - compact opstring (uncolored, values = 2*b)
!   nh_in                 - number of operators
!   bsites_in(2, nb_in)   - bond-to-site mapping (1-indexed)
!   nn_in                 - number of sites
!   nb_in                 - number of bonds
!
! Outputs:
!   parity_prefix_out(nh_in) - prefix parity at each position (int8: 0 or 1)
!   deltaK_prefix_out(nh_in) - delta K at each step (int32: -1, 0, or +1)
!   K_out                    - total loop count after all operators
!------------------------------------------!
  integer, intent(in) :: nh_in
  integer, intent(in) :: ops_compact(nh_in)
  integer, intent(in) :: nb_in
  integer, intent(in) :: bsites_in(2, nb_in)
  integer, intent(in) :: nn_in
  integer(1), intent(out) :: parity_prefix_out(nh_in)
  integer, intent(out) :: deltaK_prefix_out(nh_in)
  integer, intent(out) :: K_out

  integer :: t, b, delta_k, prefix_parity

  if (nh_in <= 0) then
     K_out = nn_in
     return
  endif

  ! Set module-level parameters
  mm_w = nh_in
  nn_w = nn_in
  nb_w = nb_in

  ! Allocate work arrays
  call alloc_work()

  ! Copy bsites
  bsites_w(:,:) = bsites_in(:,:)

  ! Initialize empty state
  opstring_w(:) = 0
  call init_empty_vertex_graph()

  prefix_parity = 0
  K_out = nn_in  ! Start with all free spins

  ! Incrementally insert operators and track parity
  ! Order matches original: update_insert → write opstring → delta_loops_insert
  do t = 1, nh_in
     b = ops_compact(t) / 2

     call do_update_insert(t-1, b)
     opstring_w(t-1) = ops_compact(t)
     call do_delta_loops_insert(t-1, b, delta_k)

     K_out = K_out + delta_k
     if (delta_k == 0) prefix_parity = 1 - prefix_parity

     parity_prefix_out(t) = int(prefix_parity, 1)
     deltaK_prefix_out(t) = delta_k
  enddo

  ! Work arrays persist for reuse across calls

end subroutine compute_parity_prefix_impl

!------------------------------------------!
! Internal helper subroutines
!------------------------------------------!

subroutine alloc_work()
  ! Only reallocate when dimensions grow beyond current allocation
  logical :: need_mm, need_nn

  need_mm = (mm_w > mm_alloc)
  need_nn = (nn_w > nn_alloc) .or. (nb_w > nb_alloc)

  if (need_mm) then
     if (allocated(opstring_w))  deallocate(opstring_w)
     if (allocated(vertexlist_w)) deallocate(vertexlist_w)
     if (allocated(st_prev_w))   deallocate(st_prev_w)
     if (allocated(st_next_w))   deallocate(st_next_w)
     if (allocated(st_used_w))   deallocate(st_used_w)

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
  ! Only zero the portion used by current call (mm_w <= mm_alloc)
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
! Core update routines
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
     prevp1 = -1
     nextp1 = -1
     prev1  = -1
     next1  = -1
  else
     c1 = st_cursor_w(s1)
     nextp1 = c1
     prevp1 = st_prev_pos_w(s1, c1)
     prev1  = leg_out_for_site_w(s1, prevp1)
     next1  = leg_in_for_site_w(s1, nextp1)
  endif

  call st_ensure_cursor_w(s2, pos)
  if (st_head_w(s2) == -1) then
     prevp2 = -1
     nextp2 = -1
     prev2  = -1
     next2  = -1
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

end module parity_prefix_mod


!------------------------------------------!
! C-callable wrapper (for ctypes)
!------------------------------------------!
subroutine compute_parity_prefix( &
    ops_compact, nh_in, bsites_in, nn_in, nb_in, &
    parity_prefix_out, deltaK_prefix_out, K_out)
  use parity_prefix_mod, only: compute_parity_prefix_impl
  implicit none
  integer, intent(in) :: nh_in
  integer, intent(in) :: ops_compact(nh_in)
  integer, intent(in) :: nb_in
  integer, intent(in) :: bsites_in(2, nb_in)
  integer, intent(in) :: nn_in
  integer(1), intent(out) :: parity_prefix_out(nh_in)
  integer, intent(out) :: deltaK_prefix_out(nh_in)
  integer, intent(out) :: K_out

  call compute_parity_prefix_impl(ops_compact, nh_in, bsites_in, nn_in, nb_in, &
                                  parity_prefix_out, deltaK_prefix_out, K_out)
end subroutine compute_parity_prefix
