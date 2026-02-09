!*************************************************************************
!*                                                                       *
!*     SPaiK --- A scalable pairwise kernel learning software            *
!*                                                                       *
!*     Computation of the value of the objective function J and          *
!*     the corresbonding subgradient. This file is specially desinged    *
!*     to solve pairwise kernel learning problems with stochastic LMBA   *
!*     and sGVT used within SPaiK software.                              * 
!*     (last modified 29.08.2025 by Napsu).                              *
!*                                                                       *
!*     Possible loss functions:                                          *
!*                                                                       *
!*     - KronRLS: Regularized least squares;                             *
!*     - KronLAD: Absolute value error;                                  *
!*     - hinge-loss: epsilon intensive hinge-loss;                       *
!*     - semi-squared hinge-loss: epsilon intensive hinge-loss           *
!*         with squared difference of predictions and labels;            *
!*     - squared hinge-loss: squared epsilon intensive hinge-loss        *
!*                                                                       *
!*     Possible regularization functions:                                *
!*                                                                       *
!*     - L0 -regularization;                                             *
!*     - double regularization with L1- and L0-norms.                    *
!*                                                                       *
!*     The work was financially supported by the Research Council of     *
!*     Finland (Project No. 345804 and 345805).                          *
!*                                                                       *
!*     The SPaiK software is covered by the MIT license.                 *                                                *
!*                                                                       *
!*                                                                       *
!*************************************************************************
!*
!*     Modules included:
!*
!*     obj_fun         ! Computation of the value and the subgradient of the
!*                     ! pairwise kernel learning problem
!*

MODULE obj_fun  ! Computation of the value and the subgradient of the 
                ! objective function.

    USE r_precision, ONLY : prec  ! Precision for reals.
    IMPLICIT NONE

    PUBLIC ::  &
        myf,   &     ! Computation of the value of the objective.
        myg,   &     ! Computation of the subgradient of the objective.
        gvt,   &     ! Generalized Vec Trick.
        sgvt,  &     ! Stochastic Generalized Vec Trick.
        sgvtG        ! Stochastic GVT for gradients.



CONTAINS

    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myf *                                               *
    !*                                                                      *
    !*     Computation of the value of the objective f = f1-f2.             *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myf(n,myx,f,iterm) !

        USE param, ONLY : zero,one,large  ! Parameters
        USE initpkl, ONLY : &
            batchind=>bi, &   !  batch indices 
            nbatch=>nb, &     !  size of the batch 
            epsilon, &     ! epsilon for epsilon intensive hinge-losses
            rf, &          ! switch for loss function
            ireg, &        ! switch for regularization
            rho, &         ! regularization parameter 
            rho2, &        ! double regularization parameter 
            autolambda, &  ! automated regularization parameter
            y, &           ! labels
            p              ! predictions
    
        IMPLICIT NONE
       ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: &
            myx            ! Vector of (dual) variables
!f2py depend(n) p(n)
!f2py depend(n) y(n)
    

        ! Scalar Arguments
        REAL(KIND=prec), INTENT(OUT) :: f  ! Value of the function.
        INTEGER, INTENT(IN) :: n           ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                               !   0  - Everything is ok.
                                               !  -3  - Failure in function calculations

        REAL(KIND=prec) :: ydiff,ptmp
        INTEGER :: i

        iterm = 0
        f = zero
        ydiff = zero
        ptmp = p(1)
                    
        call sgvt(p,myx) ! computes p = K myx
        
        SELECT CASE(rf) ! Select the loss function used                   
            CASE(1) ! KronRLS = L2-norm
                DO i=1,n
                    f = f + (p(i)-y(i))**2
                END DO
                f = 0.5_prec*f
                f = f/REAL(n,prec) !
                
            CASE(2) ! epsilon intensive hinge loss
                DO i=1,n
                    f = f + MAX (zero,ABS(p(i)-y(i)) - epsilon)
                END DO
                f = f/REAL(n,prec) 
        
            CASE(3) ! KronLAD = L1 norm
                DO i=1,n
                    f = f + ABS(p(i)-y(i))
                END DO
                f = f/REAL(n,prec) 
                
            CASE(4) ! semi-squared hinge loss
                DO i=1,n
                    f = f + MAX (zero,(p(i)-y(i))**2 - epsilon)
                END DO
                f = 0.5_prec*f
                f = f/REAL(n,prec) 
        
            CASE(5) ! SVM with hinge loss
                DO i=1,n
                    f = f + MAX (zero,1 - p(i)*y(i))
                END DO
                f = f/REAL(n,prec) 
            
        
            CASE(6) ! squared hinge loss
                DO i=1,n
                    f = f + MAX (zero,ABS(p(i)-y(i)) - epsilon)**2
                END DO
                f = 0.5_prec*f
                f = f/REAL(n,prec) 
            
            CASE DEFAULT !
                iterm = -3
                RETURN
        
        END SELECT
        
        if (autolambda == 1) then ! automaticly selected regularization parameter for first iteration
            autolambda = 0 !
                
            rho = f/(REAL(n,prec)**2) !
            rho2 = 1.0_prec*rho !/ (REAL(n,prec))
            if (ireg == 0) then !double regularization jos myx=0 näitä ei kai tarvita
                f = f + rho * sum(abs(myx)) + rho2/2.0_prec * dot_product(myx,p)

            else if (ireg == 1) then ! L1-norm
                f = f + rho * sum(abs(myx))

            else ! L2-norm
                f = f + rho/2.0_prec * dot_product(myx,p)
            end if    

        else ! regularization parameter from the user-specified list and for all but first iteration
            if (ireg == 0) then !double regularization
    
                f = f + rho * sum(abs(myx)) + rho2/2.0_prec * dot_product(myx,p)

            else if (ireg == 1) then ! L1-norm
                f = f + rho * sum(abs(myx))

            else ! L2-norm 
                f = f + rho/2.0_prec * dot_product(myx,p) ! rho * a^T Ka
            end if
            !print*,'The value of regularization parameter rho = ',rho
        end if     
                            
        ! Error checking.
        IF (f > large) iterm = -3  !
        IF (f < -large) iterm = -3 !

        RETURN
      
    END SUBROUTINE myf


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myg *                                               *
    !*                                                                      *
    !*     Computation of the subgradient of the objective function.        *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myg(n,myx,g,iterm)

        USE param, ONLY : zero,one,large  ! Parameters.
        USE initpkl, ONLY : &
            batchind=>bi, &   !  batch indices 
            nbatch=>nb, &     !  size of the batch 
            epsilon, &     ! epsilon for epsilon intensive hinge-losses
            rf, &          ! switch for ranking function.
            ireg, &        ! switch for regularization.
            rho, &         ! regularization parameter
            rho2, &        ! double regularization parameter
            y, &           ! labels
            p              ! predictions
        
        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: myx  ! Vector of variables.
        REAL(KIND=prec), DIMENSION(n), INTENT(OUT) :: g   ! Subgradient.

        ! Scalar Arguments
        INTEGER, INTENT(IN) :: n                        ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm                   ! Cause of termination:
                                                        !   0  - Everything is ok.
                                                        !  -3  - Failure in subgradient calculations
        ! Scalar Arguments
        INTEGER :: i,i2
        REAL(KIND=prec), DIMENSION(n) :: gtmp

        iterm = 0
        g = zero
        gtmp = zero
        
        ! Gradient evaluation. Only the batch components are nonzero.
        SELECT CASE(rf) ! Select the loss function used      
            CASE(1) ! KronRLS 
!               g = p - y
                do i2=1,nbatch
                    i=batchind(i2)
                    g(i)=p(i)-y(i)
                end do
                call sgvtG(g,g)
                g = g/REAL(n,prec) ! testi
            
            CASE(2) ! Hinge loss
                do i2=1,nbatch
                    i=batchind(i2)
                    if (ABS(p(i)-y(i)) > epsilon) then
                        if (p(i)-y(i) > zero) then
                            g(i) = 1.0_prec
                        else
                            g(i) = -1.0_prec
                        end if
                    end if    
                END DO
                call sgvtG(gtmp,g)
                g = gtmp/REAL(n,prec)
            
            CASE(3) ! KronLAD
    
                do i2=1,nbatch
                    i=batchind(i2)
                    IF (p(i)-y(i) > 0) THEN
                        g(i) = 1.0_prec
                    ELSE IF (p(i)-y(i) < 0) THEN
                        g(i) = -1.0_prec    
                    ELSE
                        g(i) = 0.0_prec    
                    END IF
                END DO
                call sgvtG(gtmp,g)
                g = gtmp/REAL(n,prec) 
    
            CASE(4) ! semi-squared Hinge loss
                DO i2=1,nbatch
                    i=batchind(i2)
                    if ((p(i)-y(i))**2 > epsilon) then
                        g(i) = p(i) - y(i)
                    end if    
                END DO
                call sgvtG(gtmp,g) !  computes gtmp = K g
                g = gtmp/REAL(n,prec)! 
    
            CASE(5) ! SVM with hinge loss 
                do i2=1,nbatch
                    i=batchind(i2)
                    if (1 - p(i)*y(i) > 0) then
                        g(i) = - y(i)
                    end if
                END DO !
                call sgvtG(gtmp,g)
                g = gtmp/REAL(n,prec)! 
                        
            CASE(6) ! squared hinge loss
                do i2=1,nbatch
                    i=batchind(i2)
!               DO i=1,n
                    if (ABS(p(i)-y(i)) > epsilon) then
                        if (p(i)-y(i) > zero) then
                            g(i) = p(i)-y(i)-epsilon
                        else
                            g(i) = p(i)-y(i)+epsilon
                        end if
                                    
                    end if                        
                END DO
                call sgvtG(gtmp,g)
                g = gtmp/REAL(n,prec)

            CASE DEFAULT !
                iterm = -3
        
        END SELECT
    
        ! Regularization
        IF (ireg == 0) THEN ! double regularization
            do i2=1,nbatch
                i=batchind(i2)
                IF (myx(i) > 0) THEN
                    g(i) = g(i) + rho + rho2*p(i)
                ELSE IF (myx(i) < 0) THEN
                    g(i) = g(i) - rho + rho2*p(i)   
                END IF
            END DO
        ELSE IF (ireg == 1) THEN ! L1-norm
            DO i2=1,nbatch
                i=batchind(i2)
                IF (myx(i) > 0) THEN
                    g(i) = g(i) + rho
                ELSE IF (myx(i) < 0) THEN
                    g(i) = g(i) - rho   
                END IF
            END DO
        ELSE ! L2-norm
            do i2=1,nbatch
                i=batchind(i2)
                g(i) = g(i) + rho * p(i)
            end do
        END IF
                    
        RETURN

    END SUBROUTINE myg


    
    

    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE gvt *                                               *
    !*                                                                      *
    !*     Generalized vec trick: Computation of product p = Ka for         *
    !*     training
    !*                                                                      *
    !************************************************************************
    
    subroutine gvt(p,a,n) 
            use initpkl, ONLY : & 
            r,s, &        !  indices r,s in R^n 
            matD,matT, &  !  kernels or feature representation for drugs and targets, 
                          !    matD in R^(m x m) and matT in R^(q x q)
            matM, &
            m,q           !  numbers of drugs and targets
        implicit none
        
        real(KIND=prec), intent(out) :: p(:)
        real(KIND=prec), intent(in) :: a(:) 
        integer, intent(in) :: n

        integer :: h,i,j,k

        matM = 0.0_prec

        do h=1,n 
            i = r(h)
            j = s(h)
            do k=1,q 
                matM(k,j)=matM(k,j)+a(h)*matT(k,i) 
            end do
        end do

        p = 0.0_prec
        do k=1,m
            do h=1,n
                i = r(h)
                j = s(h)
                p(h)=p(h)+matD(j,k)*matM(i,k) 
            end do
        end do
        
    end subroutine gvt    

    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE sgvt *                                               *
    !*                                                                      *
    !*     Generalized vec trick: Computation of product p = Ka for         *
    !*     training
    !*                                                                      *
    !************************************************************************
    
subroutine sgvt(p,a)  
    use initpkl, ONLY : & 
        batchind=>bi, &   !  batch indices 
        nbatch=>nb, &     !  size of the batch <= n
        r=>rbatch, &      !  indices r,s in R^nbatch 
        s=>sbatch, &      !  indices r,s in R^nbatch 
        matD,matT, &  !  kernels or feature representation for drugs and targets, 
                      !    matD in R^(m x m) and matT in R^(q x q)
        matM, &       !  saved trial matrix, matM in R^(q x m)
        m,q           !  numbers of drugs and targets
    use initslmba, ONLY : &
        n

    implicit none

    real(KIND=prec), intent(inout) :: p(:)
    real(KIND=prec), intent(in) :: a(:) 
    !integer, intent(in) :: n

    integer :: h,i,j,k,h2

    do h2=1,nbatch 
        j = s(h2)
        do k=1,q 
            matM(k,j)= 0.0_prec
        end do
    end do

    do h2=1,nbatch 
        h = batchind(h2)
    
        i = r(h2)
        j = s(h2) 
        !if (h>n) print*,'Something wrong with h in sGVT',h,n
        !if (i>q) print*,'Something wrong with i in sGVT',i,q
        !if (j>m) then
        !    print*,'Something wrong with j in sGVT',j,m
        !    stop
        !end if

        inner: do k=1,q 
            matM(k,j)=matM(k,j)+ a(h)*matT(k,i) 
        end do inner
        p(h)=0.0_prec
    end do
    

    outer: do k=1,m 
        do h2=1,nbatch
            h = batchind(h2)
            i = r(h2)
            j = s(h2)
            p(h)=p(h)+matD(j,k)*matM(i,k) 
        end do
    end do outer

end subroutine sgvt    

    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE sgvtG *                                             *
    !*                                                                      *
    !*     Generalized vec trick: Computation of product g = Kg for         *
    !*     training                                                         *
    !*                                                                      *
    !************************************************************************
    
subroutine sgvtG(gnew,g)  
    use initpkl, ONLY : & 
        batchind=>bi, &   !  batch indices 
        nbatch=>nb, &     !  size of the batch 
        r=>rbatch, &      !  indices r,s in R^n 
        s=>sbatch, &      !  indices r,s in R^n 
        matD,matT, &  !  kernels or feature representation for drugs and targets, 
                      !    matD in R^(m x m) and matT in R^(q x q)
        matG, &       !  saved trial matrix, matG in R^(q x m)
        m,q           !  numbers of drugs and targets

    implicit none

    real(KIND=prec), intent(inout) :: gnew(:) ! New subgradient gnew = K * g
    real(KIND=prec), intent(in) :: g(:)       ! Subgradient with respect to p

    integer :: h,i,j,k,h2

    do h2=1,nbatch 
        j = s(h2)

        do k=1,q 
            matG(k,j)= 0.0_prec
        end do
    end do

    do h2=1,nbatch 
        h = batchind(h2)
        i = r(h2)
        j = s(h2) 

        inner: do k=1,q 
            matG(k,j)=matG(k,j)+ g(h)*matT(k,i) 
        end do inner
        gnew(h)=0.0_prec
    end do

    outer: do k=1,m 
        do h2=1,nbatch
            h = batchind(h2)
            i = r(h2)
            j = s(h2)
            gnew(h)=gnew(h)+matD(j,k)*matG(i,k) 
        end do
    end do outer

end subroutine sgvtG    


END MODULE obj_fun
