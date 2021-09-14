!Written by Avishek Das, May 2021, avishek_das@berkeley.edu
!Computes optimal forces in potential with Mueller-Brown potential, Gaussian basis, optionally uses a value baseline (mcr/mcvb)
!main program, manages mpi side of things, calls subroutine "traj"
program valuebaseline
use mpi
implicit none

double precision, parameter:: lam=10000 !bias
integer, parameter:: Mx=21,My=21,Mt=21 !Number of basis is Mx*My*Mt
double precision:: coeff(Mx,My,Mt,2) !coefficients for both components of force
double precision:: omega(4),omegan(4) !4 elements are the full return, average indicator, kl div, quadratic cost
double precision:: delomega(Mx,My,Mt,2),delomegan(Mx,My,Mt,2) !gradients of force coefficients

!value function data structures !downstream assumes Mvx=Mx, Mvy=My, Mvt=Mt
integer, parameter:: Mvx=21,Mvy=21,Mvt=21 !Number of basis is Mvx*Mvy*Mvt for the value function 
double precision:: delV(Mvx,Mvy,Mvt),delVn(Mvx,Mvy,Mvt) !gradients of value function
double precision:: coeffV(Mvx,Mvy,Mvt) !parameters for value function

!MPI
integer:: Nw !number of trajectories
integer :: ierr,tag,cores,id,my_id,status(MPI_STATUS_SIZE),npercore,iexit !mpi stuff
double precision :: start,finish !keeps time

!Learning
double precision, parameter:: alpha=1,alphaV=1 !learning rates !set alphaV=0 for mcr and !=0 for mcvb

!counters
integer:: i,j,npercorecount,loopcount

!mpi stuff
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD,my_id,ierr)
call MPI_Comm_size(MPI_COMM_WORLD,cores,ierr)
Nw=60
npercore=floor(Nw*1.0d0/cores)
Nw=npercore*cores
if(my_id==0) print *, 'Nw:', Nw

!initialize coefficients
if (my_id==0) then
    !open(unit=10,file='coeff0x.txt')
    !open(unit=11,file='coeff0y.txt')
    !open(unit=12,file='coeff0V.txt')
    !do i=1,Mx
    !    do j=1,My
    !        read(10,*) coeff(i,j,:,1)
    !        read(11,*) coeff(i,j,:,2)
    !        read(12,*) coeffV(i,j,:)
    !    end do
    !end do
    open(unit=20,file='omega.txt')
    !open(unit=30,file='delomega.txt')
    open(unit=120,file='coeffxrunning.txt')
    open(unit=121,file='coeffyrunning.txt')
    open(unit=130,file='coeffVrunning.txt')
end if
coeff=0.0d0
coeffV=0.0d0
omega=0.0d0
delomega=0.0d0
delV=0.0d0

!optimization loop
do loopcount=0,500
start=MPI_Wtime()

omegan=0.0d0
delomegan=0.0d0
delVn=0.0d0

!grad descent step
coeff=coeff+alpha*delomega
coeffV=coeffV+alphaV*delV

!Broadcast the new coefficients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(coeff,Mx*My*Mt*2,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(coeffV,Mvx*Mvy*Mvt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)

!run trajectories and sum gradients
do npercorecount=1,npercore
call traj(Mx,My,Mt,Mvx,Mvy,Mvt,coeff,coeffV,lam,omega,delomega,delV,my_id,npercorecount,loopcount)

omegan=omegan+omega
delomegan=delomegan+delomega
delVn=delVn+delV
end do !end loop over all trajectories in single core

!!Send the return and gradients to the main node
if (my_id==0) then
    do i=1,cores-1
    id=i
    call MPI_Recv(omega,4,MPI_DOUBLE,id,100*i+1,MPI_COMM_WORLD,status,ierr)
    omegan=omegan+omega
    call MPI_Recv(delomega,Mx*My*Mt*2,MPI_DOUBLE,id,100*i+2,MPI_COMM_WORLD,status,ierr)
    delomegan=delomegan+delomega
    call MPI_Recv(delV,Mvx*Mvy*Mvt,MPI_DOUBLE,id,100*i+11,MPI_COMM_WORLD,status,ierr)
    delVn=delVn+delV
    end do

    !average the estimates
    omega=omegan/Nw
    delomega=delomegan/Nw    
    delV=delVn/Nw
    
else !receive from main node
    id=0
    call MPI_Send(omegan,4,MPI_DOUBLE,id,100*my_id+1,MPI_COMM_WORLD,ierr)
    call MPI_Send(delomegan,Mx*My*Mt*2,MPI_DOUBLE,id,100*my_id+2,MPI_COMM_WORLD,ierr)
    
    call MPI_Send(delVn,Mvx*Mvy*Mvt,MPI_DOUBLE,id,100*my_id+11,MPI_COMM_WORLD,ierr)

end if

!Broadcast the averaged quantities from main node
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(omega,4,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega,Mx*My*Mt*2,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(delV,Mvx*Mvy*Mvt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)

!Output quantities in main node
if (my_id==0) then

    !Output the estimator
    write(20,*) omega

    !Output the variational parameters
    if (modulo(loopcount,50)==0) then
    do i=1,Mx
        do j=1,My
            write(120,'(21(2x,E15.4))') coeff(i,j,:,1)
            write(121,'(21(2x,E15.4))') coeff(i,j,:,2)
            write(130,'(21(2x,E15.4))') coeffV(i,j,:)
        end do
    end do
    end if
    call flush
    
end if

finish=MPI_Wtime()
if (my_id==0) print *, 'Time taken is',finish-start
    
end do !end learning loop

call MPI_Finalize(ierr)

end program valuebaseline
!*************************************************************************!
!*************************************************************************!
!*************************************************************************!
!subroutine for running a single trajectory
subroutine traj(Mx,My,Mt,Mvx,Mvy,Mvt,coeff,coeffV,lam,omega,delomega,delV,my_id,npercorecount,loopcount)
implicit none

!arguments
integer, intent(in):: Mx,My,Mt,Mvx,Mvy,Mvt
double precision, intent(in):: coeff(Mx,My,Mt,2),lam,coeffV(Mvx,Mvy,Mvt)
integer, intent(in):: my_id,npercorecount,loopcount
double precision:: omega(4),delomega(Mx,My,Mt,2),delV(Mvx,Mvy,Mvt)

!trajectory parameters
integer, parameter:: steps=1500 !total number of timesteps
double precision, parameter:: tau=1d-4 !timestep
double precision, parameter:: D=1.0d0 !diffusion constant
double precision,parameter :: etavar=(2.0d0*D*tau)**0.5d0 !prefactor for gaussian noise
double precision,parameter:: pi=4.0d0*atan(1.0d0)

!force calc, gradient structures
double precision,dimension(2):: x0,x1,x2,delx !holders for current position and jump in every timestep
double precision,dimension(2):: F1,F0 !force
double precision:: delu(Mx,My,Mt,2) !contains partial derivative of force w.r.t. parameters, last rank is component
double precision:: gaussianx(Mx,2),gaussiany(My,2),gaussiant(Mt,2) !contains centers and variances of the gaussians for force basis
double precision:: q(Mx,My,Mt,2),delq(Mx,My,Mt,2) !malliavin weight and its jump in every timestep
double precision:: pot

!random number stuff
integer :: sizerand
integer, allocatable :: putrand(:)
double precision,dimension(2):: rand1,rand2,r,theta,eta

!value function structures
double precision:: V !value function
double precision:: gradV(Mvx,Mvy,Mvt) !gradient of value function
!value function uses the same gaussian basis as the force
double precision:: intgradV(Mvx,Mvy,Mvt) !time integrated gradV
double precision:: rval !return in every timestep

!counters
integer:: i,j,k,p

!seed random numbers
call random_seed(size=sizerand)
allocate(putrand(sizerand))
putrand(:)=1d6*my_id+npercorecount+1d3*loopcount
call random_seed(put=putrand)

!initialize all time integrals/averages
omega=0.0d0
delomega=0.0d0
delV=0.0d0
q=0.0d0
intgradV=0.0d0

!Define gaussian basis centers and variance
do j=1,Mt
  gaussiant(j,1)=1+(j-1)*steps*1.0d0/(Mt-1) 
end do
gaussiant(:,2)=((steps*1.0d0/(Mt-1))/2.0d0)**2 !variance

do j=1,Mx
  gaussianx(j,1)=-1.5+(j-1)*(1.5-(-1.5))/(Mx-1)
end do
gaussianx(:,2)=((1.5-(-1.5))/(Mx-1)/2.0d0)**2 !variance

do j=1,My
  gaussiany(j,1)=-0.5+(j-1)*(2-(-0.5))/(My-1)
end do
gaussiany(:,2)=((2-(-0.5))/(My-1)/2.0d0)**2 !variance

x0(1)=0.63
x0(2)=0.03
!starting trajectory loop
x2=x0
do i=1,steps
        
    x1=x2
    
    !generate random noise using Box Mueller algo
    if (mod(i,2)==1) then
        call random_number(rand1)
        call random_number(rand2)
        r=(-2.0d0*log(rand2))**0.5d0
        theta=2.0d0*pi*rand1
        rand1=r*cos(theta)
        rand2=r*sin(theta)
        eta=rand1
    else
        eta=rand2
    end if

    !force calc
    call force(Mx,My,Mt,Mvx,Mvy,Mvt,x1,i,coeff,coeffV,F1,F0,V,delu,gradV,gaussianx,gaussiany,gaussiant,steps,tau,pot)

    !propagate position
    delx=tau*F1+etavar*eta
    x2=x1+delx
    
    !propagate integrals (malliavin weight and the value function gradient integral)
    delq(:,:,:,1)=etavar*eta(1)*delu(:,:,:,1)/(2.0d0*D)
    delq(:,:,:,2)=etavar*eta(2)*delu(:,:,:,2)/(2.0d0*D)
    q=q+delq
    intgradV=intgradV+tau*gradV
    
    !propagate average return and return gradient integrals
    rval=sum(((delx/tau-F1)**2-(delx/tau-F0)**2)/(4.0d0*D))
    omega(3)=omega(3)+rval
    delomega=delomega+rval *q -delq/tau *V
    delV=delV+rval*intgradV-gradV*V
    
end do !trajectory propagation ends

!making estimator and gradients dimensionally correct
omega(3)=omega(3)*tau
delomega=delomega*tau
delV=delV*tau

!impose endpoint biasing
!call force to check endpoint potential
call force(Mx,My,Mt,Mvx,Mvy,Mvt,x2,i,coeff,coeffV,F1,F0,V,delu,gradV,gaussianx,gaussiany,gaussiant,steps,tau,pot)
!impose endpoint biasing
if (x2(2)>0.7d0 .and. pot<-145) then !reached target region
    omega(2)=1 !indicator function is 1
end if

!Use quadratic cost for initialization
rval=-lam*(-0.5-x2(1))**2-lam*(1.5-x2(2))**2
omega(4)=rval
delomega=delomega+rval*q
delV=delV+rval*intgradV
omega(1)=omega(3)+omega(4)

!Or, comment out quadratic cost part and uncomment following indicator observable block for final optimization
!rval=lam*omega(2)
!delomega=delomega+rval*q
!delV=delV+rval*intgradV
!omega(1)=omega(3)+rval

end subroutine traj
!*************************************************************************!
!*************************************************************************!
!*************************************************************************!
!subroutine for force calculation using gaussian basis
subroutine force(Mx,My,Mt,Mvx,Mvy,Mvt,x1,stepcount,coeff,coeffV,F1,F0,V,delu,gradV,gaussianx,gaussiany,gaussiant,steps,tau,pot)
implicit none

integer, intent(in):: Mx,My,Mt,Mvx,Mvy,Mvt,stepcount,steps !!assumes Mvx,Mvt=Mx,Mt
double precision, intent(in):: coeff(Mx,My,Mt,2), coeffV(Mvx,Mvy,Mvt),x1(2),tau
double precision, intent(in):: gaussianx(Mx,2),gaussiany(My,2),gaussiant(Mt,2)
double precision:: F1(2),F0(2),V, delu(Mx,My,Mt,2),gradV(Mvx,Mvy,Mvt)
integer:: i,j,k

double precision:: Acap(4),a(4),b(4),c(4),xc(4),yc(4)
double precision:: expfac(4), pot

delu=0.0d0
gradV=0.d0
F1=0.0d0
F0=0.0d0
V=0.0d0

Acap(1)=-200
Acap(2)=-100
Acap(3)=-170
Acap(4)=15

a(1)=-1
a(2)=-1
a(3)=-6.5
a(4)=0.7

b(1)=0
b(2)=0
b(3)=11
b(4)=0.6

c(1)=-10
c(2)=-10
c(3)=-6.5
c(4)=0.7

xc(1)=1
xc(2)=0
xc(3)=-0.5
xc(4)=-1

yc(1)=0
yc(2)=0.5
yc(3)=1.5
yc(4)=1

expfac=Acap*exp(a*(x1(1)-xc)**2+b*(x1(1)-xc)*(x1(2)-yc)+c*(x1(2)-yc)**2)
pot=sum(expfac)
F0(1)=-sum(expfac*(2.0d0*a*(x1(1)-xc)+b*(x1(2)-yc)))
F0(2)=-sum(expfac*(2.0d0*c*(x1(2)-yc)+b*(x1(1)-xc)))

do i=1,Mx
  do j=1,My
    do k=1,Mt
        !For the (i,j,k)th Gaussian, the individual terms in the force gradient and value function gradient...
        delu(i,j,k,1)=exp(-(x1(1)-gaussianx(i,1))**2/(2.0d0*gaussianx(i,2))&
        &   -(x1(2)-gaussiany(j,1))**2/(2.0d0*gaussiany(j,2))-(stepcount-gaussiant(k,1))**2/(2.0d0*gaussiant(k,2)))
        delu(i,j,k,2)=delu(i,j,k,1)
        gradV(i,j,k)=delu(i,j,k,1)

        !contribution to full force and full value function
        F1=F1+coeff(i,j,k,:)*delu(i,j,k,:)
        V=V+coeffV(i,j,k)*gradV(i,j,k)
    end do
  end do
end do

F1=F1+F0

end subroutine force
!*************************************************************************!
!*************************************************************************!
!*************************************************************************!
