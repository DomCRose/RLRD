!Written by Avishek Das, May 2021, avishek_das@berkeley.edu
!Computes optimal forces in softened Brownian bridge, Gaussian basis, optionally uses a value baseline (mcr/mcvb)
!main program, manages mpi side of things, calls subroutine "traj"
program valuebaseline
use mpi
implicit none

double precision, parameter:: lam=100 !bias
integer, parameter:: Mx=31,Mt=21 !Number of basis is Mx*Mt !Changing Mx will require downstream changes in basis definition
double precision:: coeff(Mx,Mt) !coefficients for force
double precision:: omega(3),omegan(3) !3 elements are the full return and the two terms separately, indicator and kl divergence
double precision:: delomega(Mx,Mt),delomegan(Mx,Mt) !gradients of force coefficients

!value function data structures !downstream assumes Mvx=Mx, Mvt=Mt
integer, parameter:: Mvx=31,Mvt=21 !Number of basis is Mvx*Mvt for the value function !Changing Mvx will equire downstream changes
double precision:: delV(Mvx,Mvt),delVn(Mvx,Mvt) !gradients of value function
double precision:: coeffV(Mvx,Mvt) !parameters for value function

!MPI
integer:: Nw !number of trajectories
integer :: ierr,tag,cores,id,my_id,status(MPI_STATUS_SIZE),npercore,iexit !mpi stuff
double precision :: start,finish !keeps time

!Learning
double precision, parameter:: alpha=0.4,alphaV=50.0 !learning rates !set alphaV=0 for mcr and !=0 for mcvb

!counters
integer:: i,j,npercorecount,loopcount

!mpi stuff
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD,my_id,ierr)
call MPI_Comm_size(MPI_COMM_WORLD,cores,ierr)
Nw=12
npercore=floor(Nw*1.0d0/cores)
Nw=npercore*cores
if(my_id==0) print *, 'Nw:', Nw

!initialize coefficients
if (my_id==0) then
    !open(unit=10,file='coeff0.txt')
    !do i=1,Mx
    !    read(10,*) coeff(i,:)
    !end do
    open(unit=20,file='omega.txt')
    !open(unit=30,file='delomega.txt')
    open(unit=120,file='coeffrunning.txt')
    open(unit=130,file='coeffVrunning.txt')
end if
coeff=0.0d0
coeffV=0.0d0
omega=0.0d0
delomega=0.0d0
delV=0.0d0

!optimization loop
do loopcount=0,1000
start=MPI_Wtime()

omegan=0.0d0
delomegan=0.0d0
delVn=0.0d0

!grad descent step
coeff=coeff+alpha*delomega
coeffV=coeffV+alphaV*delV

!Broadcast the new coefficients
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(coeff,Mx*Mt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(coeffV,Mvx*Mvt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)

!run trajectories and sum gradients
do npercorecount=1,npercore
call traj(Mx,Mt,Mvx,Mvt,coeff,coeffV,lam,omega,delomega,delV,my_id,npercorecount,loopcount)

omegan=omegan+omega
delomegan=delomegan+delomega
delVn=delVn+delV
end do !end loop over all trajectories in single core

!!Send the return and gradients to the main node
if (my_id==0) then
    do i=1,cores-1
    id=i
    call MPI_Recv(omega,3,MPI_DOUBLE,id,100*i+1,MPI_COMM_WORLD,status,ierr)
    omegan=omegan+omega
    call MPI_Recv(delomega,Mx*Mt,MPI_DOUBLE,id,100*i+2,MPI_COMM_WORLD,status,ierr)
    delomegan=delomegan+delomega
    call MPI_Recv(delV,Mvx*Mvt,MPI_DOUBLE,id,100*i+11,MPI_COMM_WORLD,status,ierr)
    delVn=delVn+delV
    end do

    !average the estimates
    omega=omegan/Nw
    delomega=delomegan/Nw    
    delV=delVn/Nw
    
else !receive from main node
    id=0
    call MPI_Send(omegan,3,MPI_DOUBLE,id,100*my_id+1,MPI_COMM_WORLD,ierr)
    call MPI_Send(delomegan,Mx*Mt,MPI_DOUBLE,id,100*my_id+2,MPI_COMM_WORLD,ierr)
    
    call MPI_Send(delVn,Mvx*Mvt,MPI_DOUBLE,id,100*my_id+11,MPI_COMM_WORLD,ierr)

end if

!Broadcast the averaged quantities from main node
call MPI_Barrier(MPI_COMM_WORLD,ierr)
call MPI_Bcast(omega,3,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(delomega,Mx*Mt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Bcast(delV,Mvx*Mvt,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
call MPI_Barrier(MPI_COMM_WORLD,ierr)

!Output quantities in main node
if (my_id==0) then

    !Output the estimator
    write(20,*) omega

    !Output the variational parameters
    if (modulo(loopcount,20)==0) then
    do i=1,Mx
        write(120,'(21(2x,E15.4))') coeff(i,:)
        write(130,'(21(2x,E15.4))') coeffV(i,:)
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
subroutine traj(Mx,Mt,Mvx,Mvt,coeff,coeffV,lam,omega,delomega,delV,my_id,npercorecount,loopcount)
implicit none

!arguments
integer, intent(in):: Mx,Mt,Mvx,Mvt
double precision, intent(in):: coeff(Mx,Mt),lam,coeffV(Mvx,Mvt)
integer, intent(in):: my_id,npercorecount,loopcount
double precision:: omega(3),delomega(Mx,Mt),delV(Mvx,Mvt)

!trajectory parameters
integer, parameter:: steps=1d3 !total number of timesteps
double precision, parameter:: tau=1d-3 !timestep
double precision, parameter:: D=1.0d0 !diffusion constant
double precision, parameter:: x0=0.0d0,xf=1.0d0 !starting and ending point of bridge
double precision,parameter :: etavar=(2.0d0*D*tau)**0.5d0 !prefactor for gaussian noise
double precision,parameter:: pi=4.0d0*atan(1.0d0)

!force calc, gradient structures
double precision:: x1,x2,delx !holders for current position and jump in every timestep
double precision:: F1 !force
double precision:: delu(Mx,Mt) !contains partial derivative of force w.r.t. parameters
double precision, parameter:: xleft=-4.0d0,xright=5.0d0 !grid range for centers of gaussians, xright>xf+0.5>xf>x0>x0-0.5>xleft order must be maintained
double precision:: gaussianx(Mx,2),gaussiant(Mt,2) !contains centers and variances of the gaussians for force basis
double precision:: q(Mx,Mt),delq(Mx,Mt) !malliavin weight and its jump in every timestep
double precision:: diff !needed for defining basis

!random number stuff
integer :: sizerand
integer, allocatable :: putrand(:)
double precision:: rand1,rand2,r,theta,eta

!value function structures
double precision:: V !value function
double precision:: gradV(Mvx,Mvt) !gradient of value function
!value function uses the same gaussian basis as the force
double precision:: intgradV(Mvx,Mvt) !time integrated gradV
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
  gaussiant(j,1)=1+(j-1)*steps/(Mt-1) !1,1+2sigamt,1+4sigmat,...,1+2(Mt-1)sigmat which is steps+1
end do
gaussiant(:,2)=((steps/(Mt-1))/2.0d0)**2 !variance

!define variances
!if x grid doesn't have 31 gaussians, this part will change...
gaussianx(1:11,2)=((x0-0.5-xleft)/10/2)**2
gaussianx(21:31,2)=((xright-xf-0.5)/10/2)**2
gaussianx(12:20,2)=((xf-x0+1.0)/(9+1)/2)**2

!define centers
diff=(x0-0.5-xleft)/10
do i=1,11
  gaussianx(i,1)=xleft+(i-1)*diff
end do
diff=(xf-x0+1.0)/(9+1)
do i=1,9
  gaussianx(11+i,1)=x0-0.5+i*diff
end do
diff=(xright-xf-0.5)/10
do i=1,11
  gaussianx(20+i,1)=xf+0.5+(i-1)*diff
end do

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
    call force(Mx,Mt,Mvx,Mvt,x1,i,coeff,coeffV,F1,V,delu,gradV,gaussianx,gaussiant,steps,tau)

    !propagate position
    delx=tau*F1+etavar*eta
    x2=x1+delx
    
    !propagate integrals (malliavin weight and the value function gradient integral)
    delq=etavar*eta*delu/(2.0d0*D)
    q=q+delq
    intgradV=intgradV+tau*gradV
    
    !propagate average return and return gradient integrals
    rval=((delx/tau-F1)**2-(delx/tau)**2)/(4.0d0*D)
    omega(3)=omega(3)+rval
    delomega=delomega+rval *q -delq/tau *V
    delV=delV+rval*intgradV-gradV*V
    
end do !trajectory propagation ends

!making estimator and gradients dimensionally correct
omega(3)=omega(3)*tau
delomega=delomega*tau
delV=delV*tau

!impose endpoint biasing
if (abs(x2-xf)<1d-1) then !reached target region
    omega(2)=+lam*1 !indicator function is 1
    delomega=delomega+omega(2)*q 
    delV=delV+omega(2)*intgradV
end if
omega(1)=omega(2)+omega(3)

end subroutine traj
!*************************************************************************!
!*************************************************************************!
!*************************************************************************!
!subroutine for force calculation using gaussian basis
subroutine force(Mx,Mt,Mvx,Mvt,x1,stepcount,coeff,coeffV,F1,V,delu,gradV,gaussianx,gaussiant,steps,tau)
implicit none

integer, intent(in):: Mx,Mt,Mvx,Mvt,stepcount,steps !!assumes Mvx,Mvt=Mx,Mt
double precision, intent(in):: coeff(Mx,Mt), coeffV(Mvx,Mvt),x1,tau
double precision, intent(in):: gaussianx(Mx,2),gaussiant(Mt,2)
double precision:: F1,V, delu(Mx,Mt),gradV(Mvx,Mvt)
integer:: i,j

delu=0.0d0
gradV=0.d0
F1=0.0d0
V=0.0d0

do i=1,Mx
    do j=1,Mt
        !For the (i,j)th Gaussian, the individual terms in the force gradient and value function gradient...
        delu(i,j)=exp(-(x1-gaussianx(i,1))**2/(2.0d0*gaussianx(i,2))-(stepcount-gaussiant(j,1))**2/(2.0d0*gaussiant(j,2)))
        gradV(i,j)=delu(i,j)
        
        !contribution to full force and full value function
        F1=F1+coeff(i,j)*delu(i,j)
        V=V+coeffV(i,j)*gradV(i,j)
    end do
end do

end subroutine force
!*************************************************************************!
!*************************************************************************!
!*************************************************************************!
