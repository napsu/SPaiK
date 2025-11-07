!*************************************************************************
!*                                                                       *
!*     Initialization of parameters for SPaiK                            *
!*     (version 1.0, last modified 04.09.2025)                           *
!*                                                                       *
!*     The SPaiK software is covered by the MIT license.                 *
!*                                                                       *
!*************************************************************************
!*
!*     Modules included:
!*
!*     initpkl          ! Saved parameters for pairwise learning.
!*     initslmba        ! Initialization of SLMBA -solver.
!*

MODULE initpkl  ! Saved parameters for pairwise kernel learning.

    USE r_precision, ONLY : prec  ! Precision for reals
    IMPLICIT NONE

    ! Allocatable tables
    REAL(KIND=prec), SAVE, DIMENSION(:), allocatable :: &
        y, &                      ! Labels, y(nrecords);
        p                         ! Predicted scores, p(nrecords).
    REAL(KIND=prec), SAVE, DIMENSION(:,:), allocatable :: &
        matD,matT, &              ! Kernels or feature representation for "drugs" and "targets", 
                                  !    matD in R^(m x m) and matT in R^(q x q);
        matM,matG                 ! Saved trial matrices, matM and matG in R^(q x m).


    INTEGER, SAVE, DIMENSION(:), allocatable :: &
        r, s, &                       ! Indices, r(nrecords), s(nrecords);
        rbatch, sbatch, &             ! Indices, rbatch(nrecords), sbatch(nrecords);
        bi                            ! Batch indices, bi(nrecords).
    
    ! Other Real parameters.
    REAL(KIND=prec), SAVE :: & !
        epsilon, &                   ! Epsilon for epsilon intensive hinge-losses (from python);
        rho, &                       ! Regularization parameters.
        rho2

    ! Other integer parameters.
    INTEGER, SAVE :: & !
        m,q, &                       ! Numbers of unique "drugs" and "targets" (from python);
        ibin, &                      ! ibin = 1 with binary data (from python);
        nb, &                        ! Size of the batch, nb <= nrecords; ! miksi tämä on täällä eikä tuolla slmba:ssa?
        rf, &                        ! Switch for loss function (from python);
        ireg, &                      ! Switch for regularization (from python):
                                     !   0 - L1 + L2 -norms,   
                                     !   1 - L1 -norm,
                                     !   2 - L2 -norm; 
        autolambda                   ! Switch for automated selection of lambda.


CONTAINS


    SUBROUTINE init_pklpar()    ! User supplied subroutine for further initialization of parameters (when needed).
                                ! May be left empty.
        IMPLICIT NONE

    END SUBROUTINE init_pklpar

END MODULE initpkl


MODULE initslmba  ! Initialization of parameters for SLMBA.

    USE r_precision, ONLY : prec   ! Precision for reals.
    USE param, ONLY : zero, one, small   ! Parameters.

    IMPLICIT NONE

    ! Parameters
    INTEGER, PARAMETER :: &
        na      = 2, &             ! Maximum bundle dimension, na >= 2;
        mcu     = 15, &            ! Maximum number of stored corrections, mcu >=1;
        mcinit  = 7, &             ! Initial maximum number of stored corrections, mcu >= mcinit >= 3.
                                   ! If mcinit <= 0, the default value mcinit = 3 will be used.
                                   ! However, the value mcinit = 7 is recommented;
        inma    = 3, &             ! Selection of line search method:
                                   !   inma = 0, Armijo line search,
                                   !   inma = 1, nonmonotone Armijo line search,
                                   !   inma = 2, weak Wolfe line search,
                                   !   inma = 3, nonmonotone  weak Wolfe line search;
        mnma    = 10, &            ! Maximum number of function values used in nonmonotone line search;
        maxnin  = 20               ! Maximum number of interpolations, maxnin >= 0.
                                   ! The value maxnin = 2-20 is recommented with inma=0,
                                   ! maxnin >= 20 with inma=1 and 3, and maxnin =200 with inma=2.
                                   ! For example:
                                   !   inma = 0, maxin = 20,
                                   !   inma = 1, mnma = 20, maxin = 30,
                                   !   inma = 2, maxnin = 200,
                                   !   inma = 3, mnma=10, maxnin = 20.


    ! Real parameters (if parameter value <= 0.0 the default value of the parameter will be used).
    REAL(KIND=prec), SAVE :: &
        tolb    = -small, &       ! Tolerance for the function value, 
                                  !   - If tolb == 0 the default value -large will be used;
        tolf    = 1.0E-08_prec, & ! Tolerance for change of function values (default = 1.0E-8);
        tolf2   = -10.0_prec, &   ! Second tolerance for change of function values:
                                  !   - If tolf2 < 0 the the parameter and the corresponding termination
                                  !   criterion will be ignored (recommended with inma=1,3),
                                  !   - If tolf2 == 0 the default value 1.0E+4 will be used;
        tolg    = 1.0E-6_prec, &  ! Tolerance for the termination criterion (default = 1.0E-5);
        tolg2   = 1.0E-5_prec, &  ! Tolerance for the second termination criterion (default = 1.0E-3);
        eta     = 0.50_prec, &     ! Distance measure parameter, eta > 0:
                                  !   - If eta < 0  the default value 0.0001 will be used;
        epsl    = 0.24E+00, &     ! Line search parameter, 0 < epsl < 0.25 (default = 0.24);
        xmax    = 1000.0_prec     ! Maximum stepsize, 1 < XMAX (default = 1000).

    ! Integer parameters (if value <= 0 the default value of the parameter will be used).
    INTEGER, SAVE :: & 
        n, &                      ! Number of variables (n = nrecords).
        nbatch, &                 ! Size of the batch, nbatch < m
        maxbatch, &               ! Number of different batches;
        nth, &                    ! Number of iterations after which a new batch is selected;
        batchtype = 1, &          ! Type for selecting the batches:
                                  !     0  - Totally random batches,
                                  !     1  - Random batches such that all indices are used at least once;
        mit     = 5000, &         ! Maximun number of iterations;
        mfe     = 5000, &         ! Maximun number of function evaluations;
        mtesf   =   5, &          ! Maximum number of iterations with changes of
                                  !   function values smaller than tolf (default = 10);
        iprint  =    0, &         ! Printout specification for SLMBA:
                                  !    -1  - No printout,
                                  !     0  - Only the error messages,
                                  !     1  - The final values of the objective function
                                  !          (default used if iprint < -1),
                                  !     2  - The final values of the objective function and the
                                  !          most serious warning messages,
                                  !     3  - The whole final solution,
                                  !     4  - At each iteration values of the objective function,
                                  !     5  - At each iteration the whole solution;
        iscale  =    0            ! Selection of the scaling with SLMBA:
                                  !     0  - Scaling at every iteration with STU/UTU (default),
                                  !     1  - Scaling at every iteration with STS/STU,
                                  !     2  - Interval scaling with STU/UTU,
                                  !     3  - Interval scaling with STS/STU,
                                  !     4  - Preliminary scaling with STU/UTU,
                                  !     5  - Preliminary scaling with STS/STU,
                                  !     6  - No scaling.

    INTEGER, SAVE :: ib           ! Index for batch 

    ! Allocatable tables
    REAL(KIND=prec), SAVE, DIMENSION(:), allocatable :: &
        myx                       ! Vector of variables, myx(n)

    INTEGER, SAVE, DIMENSION(:), allocatable :: &
        batchind ! Batch indices 


CONTAINS

    SUBROUTINE defaults()  ! Default values for parameters.

        USE param, ONLY: small, large, zero, one, half
        IMPLICIT NONE

        IF (iprint < -1) iprint  = 1               ! Printout specification.
        IF (mit   <= 0) mit      = 5000            ! Maximum number of iterations.
        IF (mfe   <= 0) mfe      = n*mit           ! Maximum number of function evaluations.
        IF (tolf  <= zero) tolf  = 1.0E-08_prec    ! Tolerance for change of function values.
        IF (tolf2 == zero) tolf2 = 1.0E+04_prec    ! Second tolerance for change of function values.
        IF (tolb  == zero) tolb  = -large + small  ! Tolerance for the function value.
        IF (tolg  <= zero) tolg  = 1.0E-05_prec    ! Tolerance for the termination criterion.
        IF (tolg2  <= zero) tolg = 1.0E-03_prec    ! Tolerance for the second termination criterion.
        IF (xmax  <= zero) xmax  = 1000.0_prec     ! Maximum stepsize.
        IF (eta   <  zero) eta   = 1.0E-4_prec     ! Distance measure parameter
        IF (epsl  <= zero) epsl  = 0.24_prec       ! Line search parameter,
        IF (mtesf <= 0) mtesf    = 10              ! Maximum number of iterations with changes
                                                   ! of function values smaller than tolf.
        IF (iscale > 6 .OR. iscale < 0) iscale = 0 ! Selection of the scaling.

    END SUBROUTINE defaults

    SUBROUTINE init_slmbapar()  ! Subroutine for further initialization of parameters.
        USE initpkl, ONLY : &
            ibin, &             ! ibin = 1 with binary data (set in python);
            m                   ! number of drugs (or targets in target-wise computation of batches).  

        IMPLICIT NONE

        if (ibin == 1) then ! Smaller tolerances for binary data
            tolg    = 1.0E-8_prec  ! Tolerance for the termination criterion.
            tolg2   = 1.0E-6_prec  ! Tolerance for the second termination criterion.
        end if

        if (nbatch == m) then ! 
            maxbatch = 1          ! Number of different batches
!            nth      = 500
            !nth      = 100 
            nth      = 1000 
        else if (nbatch == m-1) then
            maxbatch = 2
            nth      = 500
        else if (nbatch == m/2) then
            maxbatch = 3
!            nth      = 250
!            nth      = 100
            nth      = 500
        else if (nbatch == m/5) then ! 20%
            maxbatch = 6
!            nth      = 100 
!            nth      = 50 
            nth      = 200 
        else if (nbatch == m/10) then ! 10%
            maxbatch = 11 
!            nth      = 50
!            nth      = 30
            nth      = 100
        else if (nbatch == m/20) then ! 5%
            maxbatch = 21
!            nth      = 25
!            nth      = 15
            nth      = 50 
        else
            maxbatch = m / nbatch + 1
            if (maxbatch < 2) maxbatch = 2
            nth      = 1000 / (maxbatch-1)
            if (nth < 50) nth = 50
        end if

    END SUBROUTINE init_slmbapar

END MODULE initslmba

