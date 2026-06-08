!*************************************************************************
!*                                                                       *
!*     SPaiK    - Scalable Pairwise Kernel Learning Software using       *
!*                stochastic inexact limited memory bundle algorithm     *
!*                (ISLMBA), stochastic generalized vec trick (sGVT),     *
!*                and kernels from RLScore (version 0.2)                             *
!*                                                                       *
!*     by Napsu Karmitsa 2026 (last modified 08.02.2026).                *
!*                                                                       *
!*     The SPaiK software is covered by the MIT license.                 *
!*                                                                       *
!*************************************************************************
!*
!*
!*     Codes included:
!*
!*     spaik.py              - Main Python file. Includes RLScore calls.
!*     pkl_utility.py        - Python utility programs.
!*     spaik.f95             - Building plock between Python and Fortran for 
!*                             pairwise learning software (this file).
!*     parameters.f95        - Parameters. Inludes modules:
!*                               - r_precision - Precision for reals,
!*                               - param - Parameters,
!*     initpkl.f95           - initialization of PKL parameters and SLMBA.
!*                             Includes modules:
!*                               - initpkl     - Initialization of parameters for learning.
!*                               - initslmba   - Initialization of SLMBA.
!*     slmba.f95             - SLMBA - stochastic inexact limited memory bundle algorithm for
!*                             nonsmooth optimization (specially modified for stochpkl).
!*     objfun.f95            - computation of the function and subgradients values and
!*                             a stochastic generalized vec trick (SGVT).
!*     subpro.f95            - subprograms for SLMBA.
!*     data.py               - reading and splitting of example data sets.
!*
!*     Makefile              - makefile: builds a shared library to allow SLMBA (Fortran95 
!*                             code) to be called from Python program spaik. Uses f2py, 
!*                             Python3.7, and requires a Fortran compiler (here gfortran) 
!*                             to be installed.
!*
!*
!*    After running the makefile (type "make"), run the program by typing
!*
!*      python3.7 spaik.py
!*
!*
!*    The parameters for optimization are set to (average good) default values. To change 
!*    them modify initslmba-module in initpkl.f95 as needed. If you do, rerun Makefile. 
!*
!*
!*     References:
!*
!*     for SPaiK, sGVT, and SLMBA:
!*
!*       N. Karmitsa, T. Pahikkala, A. Airola "Scalable pairwise kernel learning 
!*       with stochastic vec trick", 2026. 
!*
!*     for RLScore:
!* 
!*       https://github.com/aatapa/RLScore
!*
!*       T. Pahikkala, A. Airola, "Rlscore: Regularized least-squares learners", 
!*       Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.
!*
!*     for SLMBA and LMBM:
!*
!*       N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola 
!*       "Stochastic limited memory bundle algorithm for clustering in big data", 
!*       Pattern Recognition, Vol 165, 111654, 2025 (A slightly different version of the SLMBA)
!*
!*       N. Haarala, K. Miettinen, M.M. Mäkelä, "Globally Convergent Limited Memory Bundle Method  
!*       for Large-Scale Nonsmooth Optimization", Mathematical Programming, Vol. 109, No. 1,
!*       pp. 181-205, 2007. DOI 10.1007/s10107-006-0728-2.
!*
!*       M. Haarala, K. Miettinen, M.M. Mäkelä, "New Limited Memory Bundle Method for Large-Scale 
!*       Nonsmooth Optimization", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004. 
!*       DOI 10.1080/10556780410001689225.
!*
!*     for NSO:
!*
!*       A. Bagirov, N. Karmitsa, M.M. Mäkelä, "Introduction to nonsmooth optimization: theory, 
!*       practice and software", Springer, 2014.
!*
!*
!*     Acknowledgements:
!*
!*     The work was financially supported by the Research Council of Finland projects 
!*     (Projects No. #340182 and #345804 led by Tapio Pahikkala and 
!*      Projects No. #340140 and #345805 led by Antti Airola).
!* 
!*************************************************************************
!*
!*     * fmodule spaik *
!*
!*     Main Fortran program for pairwise learning software with SLMBA
!*     and sGVT.
!*
!*************************************************************************

MODULE fmodule

    USE r_precision, ONLY : prec  ! Precision for reals.
    USE slmba_mod
    USE obj_fun
    IMPLICIT NONE 

CONTAINS

    SUBROUTINE spaik(apy,ppy,score,KD,KT,MM,MG,rows,cols,loss,termination,nitfinal,nrec,m,q,batchsize,seed,h)

        USE param, ONLY : zero, one, large  ! Parameters.

        USE initpkl, ONLY : &               ! Saved PKL parameters
            rf, &                           ! Switch for loss function
            p, &                            ! Predicted scores
            y, &                            ! Scores
            matD,matT, &                    ! Drug and target kernels
            matM,matG, &                    ! Auxiliary matrices for SGVT
            r,s, &                          ! Index vectors
            rbatch, sbatch, &               ! Index vectors
            bi, &                           ! Batch indices
            init_pklpar                     ! S  Furher initialization of parameters

        USE initslmba, ONLY : &             ! Initialization of SLMBA
            n, &                            ! Number of variables, n = nrec
            mcu, &                          ! Maximum number of stored corrections
            mcinit, &                       ! Initial maximum number of stored corrections
            batchind, &                     ! Indices for target(drug)-wise batches
            nbatch, &                       ! Size of the target-wise batch 
            nth, &                          ! Number of iterations after which a new batch is selected
            maxbatch, &                     ! Maximum number of batches
            epsl, &                         ! Line search parameter
            defaults, &                     ! S  Default values for parameters
            init_slmbapar                   ! S  Further initialization of parameters

        IMPLICIT NONE
 

        integer, intent(in) :: nrec,m,q,seed(8),h 
        real(kind=prec), intent(inout) :: apy(nrec)
!f2py   intent(inout) apy
!f2py   depend(nrec) apy
        real(kind=prec), intent(inout) :: ppy(nrec),MM(q,m),MG(q,m)
!f2py   intent(inout) ppy,MM,MG
!f2py   depend(nrec) ppy
!f2py   depend(q,m) MM,MG
        real(kind=prec), intent(in) :: KD(m,m),KT(q,q)
!f2py   intent(in) KD,KT
!f2py   depend(m) KD
!f2py   depend(q) KT

        integer, intent(in) :: rows(nrec),cols(nrec)
!f2py   intent(in) rows,cols
!f2py   depend(nrec) rows,cols 

        integer, intent(inout) :: termination, nitfinal
        integer, intent(in) :: batchsize
        character (len=*), intent (in) :: loss
        real(KIND=prec), dimension(:), intent(in) :: score ! Y
        real(KIND=prec) :: &
            f 
        integer :: &
            mc, &          ! Initial maximum number of stored corrections for LMBM
            i
        INTEGER, DIMENSION(4) :: & ! 
            iout           ! Output integer parameters for SLMBA.
                           !   iout(1)   Number of used iterations.
                           !   iout(2)   Number of used function evaluations.
                           !   iout(3)   Number of used subgradient evaluations
                           !   iout(4)   Cause of termination:
                           !               1  - The problem has been solved
                           !                    with desired accuracy.
                           !               2  - Changes in function values < tolf in mtesf
                           !                    subsequent iterations.
                           !               3  - Unchanging gradient norms in null steps.
                           !               4  - Number of function calls > mfe. 
                           !               5  - Number of iterations > mit. 
                           !               7  - f < tolb. 
                           !               8  - f is not changing between batches. 
                           !               9  - Successful termination.
                           !              -1  - Two consecutive restarts. 
                           !              -2  - Number of restarts > maximum number
                           !                    of restarts. 
                           !              -3  - Failure in function or subgradient
                           !                    calculations
                           !              -4  -
                           !              -5  - Invalid input parameters.
                           !              -6  - Unspecified error.
    
        nbatch = batchsize ! Size of the batch w.r.t. targets
        n  = nrec          ! Numbers of variables in optimization
        !print*,'The size of the batch w.r.t. targets: ',nbatch 
        
        ! Switch for the loss function defined in Python
        if (loss == "RLS") then
            !print*,'Learning with KronRLS.'
            rf = 1
        else if (loss == "L1") then
            !print*,'Learning with KronLAD.'
            rf = 3
        else if (loss == "hinge-loss") then
            !print*,'Learning with hinge-loss.'
            rf = 2
        else if (loss == "semi-squared-hinge") then
            !print*,'Learning with semi-squared hinge loss.'
            rf = 4
        else if (loss == "svm-hinge") then
            !print*,'Learning with svm-hinge.'
            rf = 5
        else if (loss == "squared-hinge") then
            !print*,'Learning with squared hinge loss.'
            rf = 6
            
        else        
            print*,'Sorry, no loss function "',loss,'" coded.'
            return
        end if

        allocate(p(n),y(n)) 
        allocate(matD(m,m),matT(q,q),matM(q,m),matG(q,m),r(n),s(n),rbatch(n),sbatch(n),bi(n))
        allocate(batchind(m)) ! for drug (target)-wise batches
        matD = KD
        matT = KT
        matM = MM
        matG = MG
        r = cols+1
        s = rows+1
        p = zero
        p = ppy
        y = score

        CALL init_pklpar()

        mc = mcinit          ! Initial maximum number of stored corrections
        CALL defaults()      ! Default values for (optimization) parameters.   
        CALL init_slmbapar() ! Set the maximum number of batches and the number
                             !   of iterations after which the batch is reselected.

        !if (h >= 0) then    ! Always... 
        if (h > 0) then      ! After one epoch... 
            maxbatch = 1     !     validate after each batch
            nth = 100        !     and use at most 100 iterations 
        end if

        do i=1,m ! for drug (or target)-wise batches
            batchind(i)=i
        end do

        if (n <= 0) then
            PRINT*,'n<0'
            RETURN
        else if (epsl >= 0.25_prec) then
            PRINT*,'epsl >= 0.25'
            RETURN
        else if (mcu <= 0) then
            PRINT*,'mcu <= 0'
            RETURN
        end if

        if (nbatch > m) then
            PRINT*,'Warning: nbatch > m. Applying full batch.'
            nbatch = m
        end if

        CALL slmba(apy,n,mc,f,iout(1),iout(2),iout(3),iout(4),seed)

        termination = iout(4)         ! return termination criterion to python
        nitfinal = nitfinal + iout(1) ! return number of iterations to python  
        ppy = p
        MM = matM
        MG = matG

        deallocate(p,y)
        deallocate(matD,matT,matM,matG,r,s,rbatch,sbatch,bi)
        deallocate(batchind)

        RETURN
 
    END SUBROUTINE spaik

END MODULE fmodule
