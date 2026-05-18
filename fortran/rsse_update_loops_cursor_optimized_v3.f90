!-----------------------------------------------------------------!
! Resummation-based SSE (RSSE) for SU(N) Heisenberg Model        !
! Based on: Desai & Pujari, PRB 104, L060406 (2021)              !
!                                                                 !
! This version: incremental vertex + loop counting updates        !
!-----------------------------------------------------------------!

!--------------------!
 module configuration
!--------------------!
 save

 integer :: lx, ly, nn, nb, nh, mm
 real(8) :: surface_n, beta
 logical :: use_rsse_updates  ! Flag to use RSSE local updates vs standard SSE

 integer, allocatable :: spin(:)
 integer, allocatable :: bsites(:,:)
 integer, allocatable :: opstring(:)
 integer, allocatable :: frstspinop(:)
 integer, allocatable :: lastspinop(:)
 integer, allocatable :: vertexlist(:)
 integer :: nbins_total
 integer :: current_parity  ! Incremental parity tracking (0 or 1)

 ! Site-time linked lists for O(1)-amortized update_insert
 integer, allocatable :: st_prev(:,:)    ! (2,0:mm-1): previous position on same site
 integer, allocatable :: st_next(:,:)    ! (2,0:mm-1): next position on same site
 logical, allocatable :: st_used(:,:)    ! (2,0:mm-1): slot occupied in site-time list
 integer, allocatable :: st_head(:)      ! (1:nn): smallest position on each site, -1 if empty
 integer, allocatable :: st_tail(:)      ! (1:nn): largest  position on each site, -1 if empty
 integer, allocatable :: st_cursor(:)    ! (1:nn): first >= current query pos, else head

 end module configuration
!------------------------!

!-----------------------!
 module measurementdata
!-----------------------!
 save
 ! For sign/reweighting (needed on frustrated lattices):
 !   accumulate numerator sums  sum_t [ O(t) * sign(t) ]
 !   and also denominator sums  sum_t [ sign(t) ].
 ! For bipartite SU(2) square lattice, sign(t)=+1 always.
 real(8) :: enrg1=0.d0, enrg2=0.d0
 real(8) :: sign1=0.d0, sign2=0.d0
 real(8) :: amag2=0.d0
 real(8) :: ususc=0.d0, stiff=0.d0
 real(8) :: rhosx = 0.d0, rhosy = 0.d0
 real(8) :: signb1 = 0.d0, signb2 = 0.d0

 ! Bin accumulators over completed bins (mean and mean-square)
 ! Order: 1:-E/N, 2:C/N, 3:<sign>, 4:<m^2>, 5:rho_s(new), 6:rho_s(std), 7:X(0,0)
 real(8) :: data1(7)=0.d0, data2(7)=0.d0
 end module measurementdata
!--------------------------!

!-----------------------!
 module datasetio
!-----------------------!
! V3 binary dataset writer for training:
!  - UNCOLORED opstring (0 or 2*b)
!  - nh, K, parity
!  - NO parity_prefix (recomputed on-the-fly by training script)
!
! File format (little-endian stream):
!   char[4]  magic = "RSS3"
!   int32    version = 3
!   int32    lx, ly, nn, nb, mm
!   real64   beta, surface_n
!   Then repeated samples:
!   int32    nh
!   int32    K
!   int32    parity   (0 or 1, total parity of the configuration)
!   int32[nh] opstring_uncolored
!
 implicit none
 save
 integer, parameter :: i4 = selected_int_kind(9)
 integer, parameter :: i1 = selected_int_kind(2)
 integer, parameter :: r8 = selected_real_kind(15, 307)
 integer :: ds_unit = -1
 integer(i4) :: ds_version = 4_i4
 contains

 subroutine dataset_open(filename)
   use configuration, only: lx, ly, nn, nb, mm, beta, surface_n
   character(len=*), intent(in) :: filename
   character(len=4) :: magic

   if (ds_unit /= -1) return
   magic = 'RSS4'
   open(newunit=ds_unit, file=filename, status='replace', &
        access='stream', form='unformatted', action='write', convert='little_endian')
   write(ds_unit) magic
   write(ds_unit) ds_version
   write(ds_unit) int(lx, i4), int(ly, i4), int(nn, i4), int(nb, i4), int(mm, i4)
   write(ds_unit) real(beta, r8), real(surface_n, r8)
 end subroutine dataset_open

 subroutine dataset_write_sample_v4(nh_in, parity_in)
  use configuration, only: mm, opstring
  integer, intent(in) :: nh_in, parity_in
  integer(i4), allocatable :: ops32(:)
  integer :: i, t, b

  if (ds_unit == -1) return

  write(ds_unit) int(nh_in, i4), int(parity_in, i4)

  allocate(ops32(0:nh_in-1))
  t = 0
  do i = 0, mm-1
     if (opstring(i) /= 0) then
        b = opstring(i) / 2
        ops32(t) = int(2*b, i4)   ! uncolored token
        t = t + 1
     endif
  enddo

  if (nh_in > 0) write(ds_unit) ops32
  deallocate(ops32)
 end subroutine dataset_write_sample_v4


 subroutine dataset_close()
   if (ds_unit /= -1) then
      close(ds_unit)
      ds_unit = -1
   endif
 end subroutine dataset_close

 end module datasetio
!--------------------------!

!================================!
 program rsse_heisenberg
!================================!
 use configuration
 use datasetio, only: dataset_open, dataset_write_sample_v4
 implicit none
 integer :: i, j, nbins, msteps, isteps
 integer :: use_rsse_flag
 integer :: Kalpha
 integer :: sample_idx, save_every
 real(8), external :: ran

 character(len=512) :: outdir, fname
 integer(8) :: seed0

 open(10,file='rsse_input.in',status='old')
 read(10,*) lx, ly, beta, surface_n
 read(10,*) nbins, msteps, isteps
 read(10,*) use_rsse_flag  ! 0 = standard SSE, 1 = RSSE local updates
 close(10)
 nbins_total = nbins
 use_rsse_updates = (use_rsse_flag == 1)

 ! Read seed from seed.in for filename before initran modifies it
 open(10, file='seed.in', status='old')
 read(10,*) seed0
 close(10)

 call initran(1)
 call makelattice()
 call initconfig()

 write(*,'(A)') ' =========================================='
 write(*,'(A)') ' RSSE: Resummation-based SSE'
 write(*,'(A,I4,A,I4)') '  Lattice: ', lx, ' x ', ly
 write(*,'(A,F8.3)') '  beta = ', beta
 write(*,'(A,F8.3)') '  N = ', surface_n
 if (use_rsse_updates) then
    write(*,'(A)') '  Mode: RSSE local updates (uncolored loops)'
 else
    write(*,'(A)') '  Mode: Standard SSE (colored loops)'
 endif
 write(*,'(A)') ' =========================================='


 call linkvertices()
 ! Equilibration
 ! If RSSE updates are requested, also use RSSE during thermalization.
 ! Otherwise, use the standard SSE (diagonal + loop updates).
 do i = 1, isteps
    if (use_rsse_updates) then
       call rsse_local_update()
    else
       call diagonalupdate()
       call linkvertices()
       call loopupdate()
    endif
    call adjustcutoff(i)
 enddo
 write(*,'(A,I6,A,I6)') ' Equilibration done. M = ', mm, ', nh = ', nh

 ! Open binary dataset for CV training (one record per measurement step).
 ! Change save_every to thin the chain (e.g., 5, 10, ...).
 save_every = 1
 sample_idx = 0
 outdir = "/home/user_beiqiao/private/datafile/rsse_data/fortran3_aug_bias/3x1/beta10/train"

 write(fname,'(a,"/rsse_L",i0,"x",i0,"_beta",f0.3,"_seed",i0,"_M",i0,".bin")') &
     trim(outdir), lx, ly, beta, seed0, mm

 call dataset_open(trim(fname))
 print *, "Dataset file: ", trim(fname)

 ! Measurement phase: use RSSE local updates if requested
 do j = 1, nbins
    do i = 1, msteps
       if (use_rsse_updates) then
          ! RSSE mode: use local updates based on loop topology
          ! rsse_local_update handles both operator changes AND spin coloring
          call rsse_local_update()
          ! Note: No separate loopupdate needed - assign_spin_coloring()
          ! already samples spin colorings uniformly over all loops
       else
          ! Standard SSE mode
          call diagonalupdate()
          call linkvertices()
          call loopupdate()
       endif

       sample_idx = sample_idx + 1
       if (mod(sample_idx, save_every) == 0) then
          call dataset_write_sample_v4(nh, current_parity)
       endif
       call measure()
    enddo
    call writeresults(msteps, j)
 enddo
 call deallocateall()
 end program rsse_heisenberg
!================================!

!---------------------------!
 subroutine diagonalupdate()
!---------------------------!
! Standard SSE diagonal update (used during equilibration)
! For SU(N): H_ij = (1/N) * sum_{a,b} |a_i a_j><b_i b_j|
! Matrix element is 1/N, so weight per operator is beta*J/N = beta/N (J=1)
! But the standard SSE uses beta*nb/2 because of the 1/2 in H_ij = 1/4 - S_i.S_j
! For consistency with standard SSE (N=2), we use aprob = 0.5 * beta * nb
!---------------------------!
 use configuration; implicit none
 integer :: i, b, op
 real(8) :: aprob, dprob
 real(8), external :: ran

 ! Standard SSE acceptance (same as ssebasic.f90)
 aprob = 0.5d0 * beta * nb
 dprob = 1.d0 / (0.5d0 * beta * nb)

 do i = 0, mm-1
    op = opstring(i)
    if (op == 0) then
       b = int(ran() * nb) + 1
       if (spin(bsites(1,b)) /= spin(bsites(2,b))) then
          if (ran() * (mm - nh) <= aprob) then
             opstring(i) = 2*b
             nh = nh + 1
          endif
       endif
    elseif (mod(op, 2) == 0) then
       if (ran() <= dprob * (mm - nh + 1)) then
          opstring(i) = 0
          nh = nh - 1
       endif
    else
       b = op / 2
       spin(bsites(1,b)) = -spin(bsites(1,b))
       spin(bsites(2,b)) = -spin(bsites(2,b))
    endif
 enddo

 end subroutine diagonalupdate
!-----------------------------!

!--------------------------------!
 subroutine rsse_local_update()
!--------------------------------!
! RSSE local update with INCREMENTAL loop counting and parity tracking
! Based on Desai & Pujari PRB 104, L060406 (2021)
!
! Key optimization: delta_nl=0 corresponds to parity flip.
! We track parity incrementally instead of resampling spin coloring.
!--------------------------------!
 use configuration; implicit none
 integer :: i, b, op
 integer :: delta_nl
 integer :: old_op
 real(8) :: prob_base, prob_accept
 real(8), external :: ran

 prob_base = 0.5d0 * beta * nb

 ! Monotone sweep in pos: reset site cursors to heads.
 if (allocated(st_cursor)) st_cursor(:) = st_head(:)

 ! Sweep through operator string
 do i = 0, mm-1
    op = opstring(i)

    if (op == 0) then
       ! Try insertion at empty slot
       b = int(ran() * nb) + 1

       ! Incremental vertex update + opstring insert
       call update_insert(i, b)
       opstring(i) = 2*b

       ! Compute delta_nl via loop traversal (O(loop_length))
       call delta_loops_insert(i, b, delta_nl)

       ! RSSE acceptance probability
       prob_accept = prob_base / dble(mm - nh) * (surface_n ** delta_nl)

       if (ran() < prob_accept) then
          nh = nh + 1
          ! Parity flips when delta_nl = 0
          if (delta_nl == 0) current_parity = 1 - current_parity
       else
          ! Reject: revert
          call update_remove(i)
          opstring(i) = 0
       endif

    else
       ! Try removal of existing operator
       old_op = op

       ! Compute delta_nl BEFORE removing (needs current vertex links)
       call delta_loops_remove(i, delta_nl)

       ! Incremental vertex update + opstring remove
       call update_remove(i)
       opstring(i) = 0

       ! RSSE acceptance probability
       prob_accept = dble(mm - nh + 1) / prob_base * (surface_n ** delta_nl)

       if (ran() < prob_accept) then
          nh = nh - 1
          ! Parity flips when delta_nl = 0
          if (delta_nl == 0) current_parity = 1 - current_parity
       else
          ! Reject: revert
          call update_insert(i, old_op/2)
          opstring(i) = old_op
       endif
    endif
 enddo

 end subroutine rsse_local_update
!----------------------------------!




!----------------------------------!
subroutine count_loops_fast(nl_total)
!----------------------------------!
! Fast loop counting used in the RSSE local update.
! Counts the total number of loopupdate-loops (including free spins),
!
! This keeps the high-frequency RSSE proposal path light-weight.
 use configuration; implicit none
 integer, intent(out) :: nl_total
 integer :: v0, v1, v2, s1
 integer, allocatable :: visited(:)

 allocate(visited(0:4*mm-1))
 visited(:) = 0
 nl_total = 0

 do v0 = 0, 4*mm-1, 2
    if (vertexlist(v0) < 0) cycle
    if (visited(v0) == 1) cycle
    nl_total = nl_total + 1
    v1 = v0
    do
       visited(v1) = 1
       v2 = ieor(v1, 1)
       visited(v2) = 1
       v1 = vertexlist(v2)
       if (v1 == v0) exit
    enddo
 enddo

 do s1 = 1, nn
    if (frstspinop(s1) == -1) nl_total = nl_total + 1
 enddo

 deallocate(visited)
end subroutine count_loops_fast
!----------------------------------!



!-------------------------!
 subroutine linkvertices()
!-------------------------!
 use configuration; implicit none
 integer :: b, op, s1, s2, v0, v1, v2

 frstspinop(:) = -1
 lastspinop(:) = -1
 call st_allocate()

 do v0 = 0, 4*mm-1, 4
    op = opstring(v0/4)
    if (op /= 0) then
       b = op / 2
       s1 = bsites(1, b)
       s2 = bsites(2, b)
       v1 = lastspinop(s1)
       v2 = lastspinop(s2)
       if (v1 /= -1) then
          vertexlist(v1) = v0
          vertexlist(v0) = v1
       else
          frstspinop(s1) = v0
       endif
       if (v2 /= -1) then
          vertexlist(v2) = v0 + 1
          vertexlist(v0 + 1) = v2
       else
          frstspinop(s2) = v0 + 1
       endif
       lastspinop(s1) = v0 + 2
       lastspinop(s2) = v0 + 3

       call st_append_site(s1, v0/4, 1)
       call st_append_site(s2, v0/4, 2)
    else
       vertexlist(v0:v0+3) = -1
    endif
 enddo

 do s1 = 1, nn
    v1 = frstspinop(s1)
    if (v1 /= -1) then
       v2 = lastspinop(s1)
       vertexlist(v2) = v1
       vertexlist(v1) = v2
    endif
 enddo

 ! For monotone queries, start cursors from the current heads.
 st_cursor(:) = st_head(:)

 end subroutine linkvertices
!---------------------------!

!---------------------------!
 subroutine st_allocate()
!---------------------------!
 use configuration; implicit none
 logical :: need_realloc

 need_realloc = .false.

 if (.not. allocated(st_prev)) then
    need_realloc = .true.
 else
    if (size(st_prev,1) /= 2) need_realloc = .true.
    if (lbound(st_prev,2) /= 0) need_realloc = .true.
    if (ubound(st_prev,2) /= mm-1) need_realloc = .true.
    if (.not. allocated(st_head)) then
       need_realloc = .true.
    else if (size(st_head) /= nn) then
       need_realloc = .true.
    endif
 endif

 if (need_realloc) then
    if (allocated(st_prev))   deallocate(st_prev)
    if (allocated(st_next))   deallocate(st_next)
    if (allocated(st_used))   deallocate(st_used)
    if (allocated(st_head))   deallocate(st_head)
    if (allocated(st_tail))   deallocate(st_tail)
    if (allocated(st_cursor)) deallocate(st_cursor)

    allocate(st_prev(2,0:mm-1))
    allocate(st_next(2,0:mm-1))
    allocate(st_used(2,0:mm-1))
    allocate(st_head(nn))
    allocate(st_tail(nn))
    allocate(st_cursor(nn))
 endif

 call st_reset_empty()
 end subroutine st_allocate

!---------------------------!
 subroutine st_reset_empty()
!---------------------------!
 use configuration; implicit none
 if (.not. allocated(st_prev)) return
 st_prev(:,:) = -1
 st_next(:,:) = -1
 st_used(:,:) = .false.
 st_head(:)   = -1
 st_tail(:)   = -1
 st_cursor(:) = -1
 end subroutine st_reset_empty

!---------------------------!
 integer function st_slot_of_site_occupied(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: b
 b = opstring(p) / 2
 if (bsites(1,b) == site) then
    st_slot_of_site_occupied = 1
 else
    st_slot_of_site_occupied = 2
 endif
 end function st_slot_of_site_occupied

!---------------------------!
 integer function st_slot_of_site_bond(site, bond_b)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, bond_b
 if (bsites(1,bond_b) == site) then
    st_slot_of_site_bond = 1
 else
    st_slot_of_site_bond = 2
 endif
 end function st_slot_of_site_bond

!---------------------------!
 integer function st_prev_pos(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: slot
 integer :: st_slot_of_site_occupied
 slot = st_slot_of_site_occupied(site, p)
 st_prev_pos = st_prev(slot, p)
 end function st_prev_pos

!---------------------------!
 integer function st_next_pos(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: slot
 integer :: st_slot_of_site_occupied
 slot = st_slot_of_site_occupied(site, p)
 st_next_pos = st_next(slot, p)
 end function st_next_pos

!---------------------------!
 subroutine st_append_site(site, p, slot)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p, slot
 integer :: headp, tailp, tail_slot, head_slot
 integer :: st_slot_of_site_occupied

 if (st_head(site) == -1) then
    st_head(site) = p
    st_tail(site) = p
    st_prev(slot,p) = p
    st_next(slot,p) = p
    st_used(slot,p) = .true.
 else
    headp = st_head(site)
    tailp = st_tail(site)
    tail_slot = st_slot_of_site_occupied(site, tailp)
    head_slot = st_slot_of_site_occupied(site, headp)

    st_prev(slot,p) = tailp
    st_next(slot,p) = headp
    st_used(slot,p) = .true.

    st_next(tail_slot, tailp) = p
    st_prev(head_slot, headp) = p

    st_tail(site) = p
 endif
 end subroutine st_append_site

!---------------------------!
 subroutine st_ensure_cursor(site, q)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, q
 integer :: c, n
 integer :: st_next_pos

 if (st_head(site) == -1) then
    st_cursor(site) = -1
    return
 endif

 c = st_cursor(site)
 if (c == -1) c = st_head(site)

 do while (c < q)
    if (c == st_tail(site)) then
       c = st_head(site)
       exit
    else
       n = st_next_pos(site, c)
       c = n
    endif
 enddo

 st_cursor(site) = c
 end subroutine st_ensure_cursor

!---------------------------!
 subroutine st_insert_site(site, p, slot, prevp, nextp)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p, slot, prevp, nextp
 integer :: prev_slot, next_slot
 integer :: st_slot_of_site_occupied

 if (st_head(site) == -1) then
    st_head(site) = p
    st_tail(site) = p
    st_cursor(site) = p
    st_prev(slot,p) = p
    st_next(slot,p) = p
    st_used(slot,p) = .true.
    return
 endif

 prev_slot = st_slot_of_site_occupied(site, prevp)
 next_slot = st_slot_of_site_occupied(site, nextp)

 st_prev(slot,p) = prevp
 st_next(slot,p) = nextp
 st_used(slot,p) = .true.

 st_next(prev_slot, prevp) = p
 st_prev(next_slot, nextp) = p

 if (p < st_head(site)) st_head(site) = p
 if (p > st_tail(site)) st_tail(site) = p
 st_cursor(site) = p
 end subroutine st_insert_site

!---------------------------!
 subroutine st_remove_site(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: slot, pp, np, pslot, nslot
 integer :: st_slot_of_site_occupied

 if (st_head(site) == -1) return

 slot = st_slot_of_site_occupied(site, p)
 pp   = st_prev(slot, p)
 np   = st_next(slot, p)

 if (pp == p .and. np == p) then
    st_head(site) = -1
    st_tail(site) = -1
    if (st_cursor(site) == p) st_cursor(site) = -1
 else
    pslot = st_slot_of_site_occupied(site, pp)
    nslot = st_slot_of_site_occupied(site, np)
    st_next(pslot, pp) = np
    st_prev(nslot, np) = pp
    if (st_head(site) == p) st_head(site) = np
    if (st_tail(site) == p) st_tail(site) = pp
    if (st_cursor(site) == p) st_cursor(site) = np
 endif

 st_prev(slot,p) = -1
 st_next(slot,p) = -1
 st_used(slot,p) = .false.
 end subroutine st_remove_site

!---------------------------!
 integer function leg_in_for_site(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: b
 b = opstring(p) / 2
 if (bsites(1,b) == site) then
    leg_in_for_site = 4*p
 else
    leg_in_for_site = 4*p + 1
 endif
 end function leg_in_for_site

!---------------------------!
 integer function leg_out_for_site(site, p)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, p
 integer :: b
 b = opstring(p) / 2
 if (bsites(1,b) == site) then
    leg_out_for_site = 4*p + 2
 else
    leg_out_for_site = 4*p + 3
 endif
 end function leg_out_for_site

!---------------------------!
 subroutine update_insert(pos, bond_b)
!---------------------------!
! Incremental vertex update for operator insertion.
! Uses site-time linked lists + per-site cursor for O(1)-amortized
! predecessor/successor queries under monotone sweeps in pos.
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: pos, bond_b
 integer :: s1, s2, v0
 integer :: c1, c2, prevp1, nextp1, prevp2, nextp2
 integer :: prev1, next1, prev2, next2
 integer :: st_prev_pos, leg_out_for_site, leg_in_for_site

 v0 = 4*pos
 s1 = bsites(1, bond_b)
 s2 = bsites(2, bond_b)

 call st_ensure_cursor(s1, pos)
 if (st_head(s1) == -1) then
    prevp1 = -1
    nextp1 = -1
    prev1  = -1
    next1  = -1
 else
    c1 = st_cursor(s1)
    nextp1 = c1
    prevp1 = st_prev_pos(s1, c1)
    prev1  = leg_out_for_site(s1, prevp1)
    next1  = leg_in_for_site(s1, nextp1)
 endif

 call st_ensure_cursor(s2, pos)
 if (st_head(s2) == -1) then
    prevp2 = -1
    nextp2 = -1
    prev2  = -1
    next2  = -1
 else
    c2 = st_cursor(s2)
    nextp2 = c2
    prevp2 = st_prev_pos(s2, c2)
    prev2  = leg_out_for_site(s2, prevp2)
    next2  = leg_in_for_site(s2, nextp2)
 endif

 call insert_one_site(s1, v0, v0+2, prev1, next1)
 call insert_one_site(s2, v0+1, v0+3, prev2, next2)

 if (prevp1 == -1) then
    call st_insert_site(s1, pos, 1, -1, -1)
 else
    call st_insert_site(s1, pos, 1, prevp1, nextp1)
 endif

 if (prevp2 == -1) then
    call st_insert_site(s2, pos, 2, -1, -1)
 else
    call st_insert_site(s2, pos, 2, prevp2, nextp2)
 endif
 end subroutine update_insert

!---------------------------!
 subroutine insert_one_site(site, vin, vout, prev, next)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: site, vin, vout, prev, next
 integer :: prev_eff, next_eff
 if (frstspinop(site) == -1) then
    frstspinop(site) = vin; lastspinop(site) = vout
    vertexlist(vin) = vout; vertexlist(vout) = vin
 else
    prev_eff = prev; if (prev_eff == -1) prev_eff = lastspinop(site)
    next_eff = next; if (next_eff == -1) next_eff = frstspinop(site)
    vertexlist(prev_eff) = vin; vertexlist(vin) = prev_eff
    vertexlist(vout) = next_eff; vertexlist(next_eff) = vout
    if (prev == -1) frstspinop(site) = vin
    if (next == -1) lastspinop(site) = vout
 endif
 end subroutine insert_one_site

!---------------------------!
 subroutine update_remove_vertex_only(pos)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: pos
 integer :: s1, s2, v0, b, prev1, next1, prev2, next2
 v0=4*pos; b=opstring(pos)/2; s1=bsites(1,b); s2=bsites(2,b)
 prev1=vertexlist(v0); next1=vertexlist(v0+2)
 prev2=vertexlist(v0+1); next2=vertexlist(v0+3)
 if (prev1==v0+2) then
    frstspinop(s1)=-1; lastspinop(s1)=-1
 else
    vertexlist(prev1)=next1; vertexlist(next1)=prev1
    if (frstspinop(s1)==v0) frstspinop(s1)=next1
    if (lastspinop(s1)==v0+2) lastspinop(s1)=prev1
 endif
 if (prev2==v0+3) then
    frstspinop(s2)=-1; lastspinop(s2)=-1
 else
    vertexlist(prev2)=next2; vertexlist(next2)=prev2
    if (frstspinop(s2)==v0+1) frstspinop(s2)=next2
    if (lastspinop(s2)==v0+3) lastspinop(s2)=prev2
 endif
 vertexlist(v0:v0+3)=-1
 end subroutine update_remove_vertex_only

!---------------------------!
 subroutine update_remove(pos)
!---------------------------!
 use configuration; implicit none
 integer, intent(in) :: pos
 integer :: b, s1, s2
 b  = opstring(pos)/2
 s1 = bsites(1,b)
 s2 = bsites(2,b)
 call st_remove_site(s1, pos)
 call st_remove_site(s2, pos)
 call update_remove_vertex_only(pos)
 end subroutine update_remove

!------------------------------------------!
 subroutine delta_loops_insert(pos, bond_b, delta_k)
!------------------------------------------!
! Incremental loop count change for operator insertion.
! Must be called AFTER update_insert + opstring(pos)=2*bond_b,
! so the vertex graph already reflects the inserted operator.
!
! Returns delta_k = change in number of loops.
!------------------------------------------!
 use configuration; implicit none
 integer, intent(in) :: pos, bond_b
 integer, intent(out) :: delta_k
 integer :: s1, s2, v0, prev1, next1, prev2, next2, v1, v2, first_hit

 v0 = 4*pos
 s1 = bsites(1, bond_b)
 s2 = bsites(2, bond_b)
 prev1 = vertexlist(v0)
 next1 = vertexlist(v0+2)
 prev2 = vertexlist(v0+1)
 next2 = vertexlist(v0+3)

 ! Special case: site was free spin (self-loop)
 if (prev1 == v0+2 .or. prev2 == v0+3) then
    delta_k = -1
    return
 endif

 ! General case: traverse loop from v0 to find which target we hit first
 v1 = v0
 do
    v2 = vertexlist(v1)
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

 end subroutine delta_loops_insert
!------------------------------------------!

!------------------------------------------!
 subroutine delta_loops_remove(pos, delta_k)
!------------------------------------------!
! Incremental loop count change for operator removal.
! Must be called BEFORE update_remove, so the vertex graph
! still contains the operator to be removed.
!
! Returns delta_k = change in number of loops.
!------------------------------------------!
 use configuration; implicit none
 integer, intent(in) :: pos
 integer, intent(out) :: delta_k
 integer :: s1, s2, b, v0, prev1, next1, prev2, next2, v1, v2, first_hit

 v0 = 4*pos
 b = opstring(pos) / 2
 s1 = bsites(1, b)
 s2 = bsites(2, b)
 prev1 = vertexlist(v0)
 next1 = vertexlist(v0+2)
 prev2 = vertexlist(v0+1)
 next2 = vertexlist(v0+3)

 ! Special case: site becomes free spin
 if (prev1 == v0+2 .or. prev2 == v0+3) then
    delta_k = 1
    return
 endif

 ! General case: traverse loop from v0 to find which target we hit first
 v1 = v0
 do
    v2 = vertexlist(v1)
    v1 = ieor(v2, 1)
    if (v1 == prev2 .or. v1 == next1 .or. v1 == next2) then
       first_hit = v1
       exit
    endif
 enddo

 if (first_hit == next1) then
    delta_k = 1
 else if (first_hit == prev2) then
    delta_k = -1
 else if (first_hit == next2) then
    delta_k = 0
 endif

 end subroutine delta_loops_remove
!------------------------------------------!

!-----------------------!
 subroutine loopupdate()
!-----------------------!
! Standard loop update - flips loops with probability 1/2
! This samples all spin colorings uniformly
!-----------------------!
 use configuration; implicit none
 integer :: i, v0, v1, v2
 real(8), external :: ran

 do v0 = 0, 4*mm-1, 2
    if (vertexlist(v0) < 0) cycle
    v1 = v0
    if (ran() < 0.5d0) then
       do
          opstring(v1/4) = ieor(opstring(v1/4), 1)
          vertexlist(v1) = -2
          v2 = ieor(v1, 1)
          v1 = vertexlist(v2)
          vertexlist(v2) = -2
          if (v1 == v0) exit
       enddo
    else
       do
          vertexlist(v1) = -1
          v2 = ieor(v1, 1)
          v1 = vertexlist(v2)
          vertexlist(v2) = -1
          if (v1 == v0) exit
       enddo
    endif
 enddo

 do i = 1, nn
    if (frstspinop(i) /= -1) then
       if (vertexlist(frstspinop(i)) == -2) spin(i) = -spin(i)
    else
       if (ran() < 0.5d0) spin(i) = -spin(i)
    endif
 enddo

 end subroutine loopupdate
!-------------------------!

!--------------------!
subroutine measure()
!--------------------!
  use configuration
  use measurementdata
  implicit none

  real(8) :: sgn

  ! ---- sign from incremental parity tracking ----
  if (current_parity == 0) then
     sgn = 1.d0
  else
     sgn = -1.d0
  endif

  ! ---- accumulate (signed) estimators ----
  sign1 = sign1 + sgn
  sign2 = sign2 + sgn*sgn

  enrg1 = enrg1 + dfloat(nh) * sgn
  enrg2 = enrg2 + dfloat(nh)**2 * sgn

  ! Note: Other observables (amag2, rhosx, rhosy, ususc) require
  ! spin coloring and are set to zero in uncolored RSSE mode
  amag2 = 0.d0
  rhosx = 0.d0
  rhosy = 0.d0
  ususc = 0.d0

end subroutine measure
!----------------------!


!------------------------------------!
subroutine writeresults(msteps, bins)
!------------------------------------!
  use configuration
  use measurementdata
  implicit none

  integer :: msteps, bins
  real(8) :: wdata1(7), wdata2(7)
  real(8) :: avg_sign, den, tiny
  real(8) :: mean_sign, std_sign
  real(8) :: nh1, nh2, am2m, usm, wx2m, wy2m
  real(8) :: rhosav

  tiny = 1.d-12

  !========================
  ! 1 Bin-average sign (for reweighting this bin)
  !========================
  avg_sign = sign1 / dble(msteps)     ! <sign> within THIS bin
  den = avg_sign

  ! Accumulate bin-averaged <sign> across bins for mean/std (bin-to-bin)
  signb1 = signb1 + avg_sign
  signb2 = signb2 + avg_sign*avg_sign
  mean_sign = signb1 / dble(bins)
  std_sign  = sqrt( abs(signb2/dble(bins) - mean_sign*mean_sign) / dble(bins) )

  !========================
  ! 2 Reweight bin means: <O> = <O*sign>/<sign>
  !========================
  if (abs(den) < tiny) then
     nh1  = 0.d0
     nh2  = 0.d0
     am2m = 0.d0
     usm  = 0.d0
     wx2m = 0.d0
     wy2m = 0.d0
  else
     nh1  = (enrg1 / dble(msteps)) / den
     nh2  = (enrg2 / dble(msteps)) / den
     am2m = (amag2 / dble(msteps)) / den
     usm  = (ususc / dble(msteps)) / den
     wx2m = (rhosx / dble(msteps)) / den
     wy2m = (rhosy / dble(msteps)) / den
  endif

  !========================
  ! 3 SSE estimators (THIS bin values)
  !========================
  wdata1(1) = nh1 / (beta * dble(nn)) - dble(nb) / (4.d0 * dble(nn))   ! -E/N
  wdata1(2) = (nh2 - nh1*(nh1 + 1.d0)) / dble(nn)                      ! C/N
  wdata1(3) = 3.d0 * am2m / dble(nn)**2                                 ! <m^2> (square only; else 0)
  wdata1(4) = 1.5d0 * wx2m / (beta * dble(nn))                           ! rho_s,x
  wdata1(5) = 1.5d0 * wy2m / (beta * dble(nn))                           ! rho_s,y
  rhosav    = 0.5d0 * (wdata1(4) + wdata1(5))
  wdata1(6) = rhosav                                                     ! rho_s,avg
  wdata1(7) = beta * usm / dble(nn)                                      ! X(0,0)

  !========================
  ! 4 Bin-to-bin mean & std for observables
  !========================
  data1(:) = data1(:) + wdata1(:)
  data2(:) = data2(:) + wdata1(:)**2

  wdata1(:) = data1(:) / dble(bins)
  wdata2(:) = data2(:) / dble(bins)
  wdata2(:) = sqrt(abs(wdata2(:) - wdata1(:)**2) / dble(bins))

  !=========================================================
  ! Output: one file
  !  - per-bin table (every call)
  !  - final human-readable summary (only at last bin)
  !=========================================================
  open(unit=10, file='rsse_results.txt', status='unknown', position='append')

  ! ---- (A) Per-bin table ----
  if (bins == 1) then
     write(10,*) '# RSSE per-bin table'
     write(10,*) ' cols: bin, -E/N,err, C/N,err, <m^2>,err, rhos_x,err, rhos_y,err, rhos_av,err, X(0,0),err'
     write(10,*) '       <sign>_mean, <sign>_err'

  endif

  write(10,'(I8, 1X, 14(1X,F14.8), 1X, F14.8, 1X, F14.8)') bins, &
       wdata1(1), wdata2(1), &
       wdata1(2), wdata2(2), &
       wdata1(3), wdata2(3), &
       wdata1(4), wdata2(4), &
       wdata1(5), wdata2(5), &
       wdata1(6), wdata2(6), &
       wdata1(7), wdata2(7), &
       mean_sign, std_sign

  ! ---- (B) Final labeled summary (only once) ----
  if (bins == nbins_total) then
     write(10,*) ' '
     write(10,*) ' =========================================='
     write(10,*) ' RSSE: Resummation-based SSE'
     write(10,'(A,I4,A,I4)') '  Lattice: ', lx, ' x ', ly
     write(10,'(A,F8.3)')    '  beta = ', beta
     write(10,'(A,F8.3)')    '  N = ', surface_n
     if (use_rsse_updates) then
        write(10,*) '  Mode: RSSE local updates (uncolored loops)'
     else
        write(10,*) '  Mode: Standard SSE (colored loops)'
     endif
     write(10,*) ' =========================================='
     write(10,*) ' '

     write(10,*) ' RSSE Results for SU(N) Heisenberg'
     write(10,*) ' Cut-off M : ', mm
     write(10,*) ' Bins completed : ', bins
     write(10,*) ' ========================================='
     write(10,'(A,2F15.8)') '  -E/N        : ', wdata1(1), wdata2(1)
     write(10,'(A,2F15.8)') '   C/N        : ', wdata1(2), wdata2(2)
     write(10,'(A,2F15.8)') '   <m**2>     : ', wdata1(3), wdata2(3)
     write(10,'(A,2F15.8)') '   rhos_x     : ', wdata1(4), wdata2(4)
     write(10,'(A,2F15.8)') '   rhos_y     : ', wdata1(5), wdata2(5)
     write(10,'(A,2F15.8)') '   rhos_av    : ', wdata1(6), wdata2(6)
     write(10,'(A,2F15.8)') '   X(0,0)     : ', wdata1(7), wdata2(7)
     write(10,'(A,2F15.8)') '   <sign>     : ', mean_sign, std_sign
     write(10,*) ' ========================================='
  endif


  close(10)



  !========================
  ! 6 Reset per-bin accumulators (NOT data1/data2/signb1/signb2)
  !========================
  enrg1 = 0.d0; enrg2 = 0.d0
  amag2 = 0.d0; ususc = 0.d0
  rhosx = 0.d0; rhosy = 0.d0
  sign1 = 0.d0; sign2 = 0.d0

end subroutine writeresults
!---------------------------!





!--------------------------!
 subroutine adjustcutoff(step)
!--------------------------!
 use configuration; implicit none
 integer, allocatable :: stringcopy(:)
 integer :: mmnew, step

 mmnew = nh + nh/3
 if (mmnew <= mm) return

 allocate(stringcopy(0:mm-1))
 stringcopy(:) = opstring(:)
 deallocate(opstring)
 allocate(opstring(0:mmnew-1))
 opstring(0:mm-1) = stringcopy(:)
 opstring(mm:mmnew-1) = 0
 deallocate(stringcopy)
 mm = mmnew

 deallocate(vertexlist)
 allocate(vertexlist(0:4*mm-1))

 ! mm changed: rebuild full working structures once.
 call linkvertices()

 end subroutine adjustcutoff
!---------------------------!

!-----------------------!
 subroutine initconfig()
!-----------------------!
 use configuration; implicit none
 integer :: i
 real(8), external :: ran

 allocate(spin(nn))
 do i = 1, nn
    spin(i) = 2*int(2.*ran()) - 1
 enddo

 mm = 20
 allocate(opstring(0:mm-1))
 opstring(:) = 0
 nh = 0
 allocate(frstspinop(nn))
 allocate(lastspinop(nn))
 allocate(vertexlist(0:4*mm-1))
 call st_allocate()
 ! Initialize parity (0 for even, 1 for odd)
 current_parity = 0

 end subroutine initconfig
!-------------------------!

!------------------------!
subroutine makelattice()
!------------------------!
  use configuration
  implicit none
  integer :: s, x1, x2, y1, y2
  integer :: Lyabs
  logical :: is_tri_pbc

  ! Convention:
  !  - Square PBC: ly > 0  (default)
  !  - Triangle PBC: ly < 0, with |ly| = L_y
  !
  ! Special test case: 3-site triangle (no 2D PBC), enabled by lx=3, ly=1.

  is_tri_pbc = (ly < 0)
  Lyabs = abs(ly)

  ! -------- 3-site triangle (no 2D PBC) --------
  if (lx == 3 .and. ly == 1) then
     nn = 3
     nb = 3
     allocate(bsites(2, nb))
     bsites(1,1) = 1; bsites(2,1) = 2
     bsites(1,2) = 2; bsites(2,2) = 3
     bsites(1,3) = 3; bsites(2,3) = 1
     return
  endif

  ! -------- Triangular lattice with PBC (with dedup) --------
  if (is_tri_pbc) then
     nn = lx * Lyabs
     ! Generate all 3*nn directed bonds into temp array, then deduplicate
     block
       integer :: raw_s1(3*nn), raw_s2(3*nn), nraw, k, s1t, s2t
       integer :: lo, hi
       logical :: dup

       nraw = 0
       do y1 = 0, Lyabs-1
       do x1 = 0, lx-1
          s = 1 + x1 + y1*lx
          ! +x
          x2 = mod(x1+1, lx); y2 = y1
          nraw = nraw + 1; raw_s1(nraw) = s; raw_s2(nraw) = 1 + x2 + y2*lx
          ! +y
          x2 = x1; y2 = mod(y1+1, Lyabs)
          nraw = nraw + 1; raw_s1(nraw) = s; raw_s2(nraw) = 1 + x2 + y2*lx
          ! +x+y
          x2 = mod(x1+1, lx); y2 = mod(y1+1, Lyabs)
          nraw = nraw + 1; raw_s1(nraw) = s; raw_s2(nraw) = 1 + x2 + y2*lx
       enddo
       enddo

       ! Deduplicate: keep first occurrence of each undirected edge
       nb = 0
       allocate(bsites(2, nraw))  ! allocate max, trim later
       do k = 1, nraw
          s1t = raw_s1(k); s2t = raw_s2(k)
          lo = min(s1t, s2t); hi = max(s1t, s2t)
          dup = .false.
          do s = 1, nb
             if (min(bsites(1,s),bsites(2,s)) == lo .and. &
                 max(bsites(1,s),bsites(2,s)) == hi) then
                dup = .true.
                exit
             endif
          enddo
          if (.not. dup) then
             nb = nb + 1
             bsites(1, nb) = s1t
             bsites(2, nb) = s2t
          endif
       enddo
       ! Note: bsites is allocated larger than nb, but only 1:nb is used.
       ! This is fine since nb is set correctly.
     end block
     return
  endif

  ! -------- Square lattice with PBC (default) --------
  nn = lx * ly
  nb = 2 * nn
  allocate(bsites(2, nb))

  do y1 = 0, ly-1
  do x1 = 0, lx-1
     s = 1 + x1 + y1*lx

     ! +x bonds
     x2 = mod(x1+1, lx)
     y2 = y1
     bsites(1, s) = s
     bsites(2, s) = 1 + x2 + y2*lx

     ! +y bonds
     x2 = x1
     y2 = mod(y1+1, ly)
     bsites(1, s+nn) = s
     bsites(2, s+nn) = 1 + x2 + y2*lx
  enddo
  enddo

end subroutine makelattice
!--------------------------!


!--------------------------!
 subroutine deallocateall()
!--------------------------!
 use configuration
 use datasetio, only: dataset_close
 implicit none
 call dataset_close()
 deallocate(spin, bsites, opstring, frstspinop, lastspinop, vertexlist)
 if (allocated(st_prev)) deallocate(st_prev, st_next, st_used, st_head, st_tail, st_cursor)
 end subroutine deallocateall
!----------------------------!

!----------------------!
 real(8) function ran()
!----------------------!
 implicit none
 real(8) :: dmu64
 integer(8) :: ran64, mul64, add64
 common/bran64/dmu64, ran64, mul64, add64
 ran64 = ran64 * mul64 + add64
 ran = 0.5d0 + dmu64 * dble(ran64)
 end function ran
!----------------!

!---------------------!
 subroutine initran(w)
!---------------------!
 implicit none
 integer(8) :: irmax
 integer(4) :: w
 real(8) :: dmu64
 integer(8) :: ran64, mul64, add64
 common/bran64/dmu64, ran64, mul64, add64

 irmax = 2_8**31
 irmax = 2*(irmax**2 - 1) + 1
 mul64 = 2862933555777941757_8
 add64 = 1013904243
 dmu64 = 0.5d0 / dble(irmax)

 open(10, file='seed.in', status='old')
 read(10,*) ran64
 close(10)
 if (w /= 0) then
    open(10, file='seed.in', status='unknown')
    write(10,*) abs((ran64*mul64)/5 + 5265361)
    close(10)
 endif
 end subroutine initran
!----------------------!
