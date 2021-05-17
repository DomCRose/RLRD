!*****************************************************************************80
!*****************************************************************************80
program srk2
implicit none

!arguments
integer,parameter:: M=4,my_id=0,walkerno=1
double precision:: a(2*M)
double precision,parameter:: kldf=0.3 !!current biasing value

!parameters
integer, parameter:: N=1
integer,parameter :: ithreshold=2 !time after which start averaging
integer(kind=8), parameter:: steps=5d6+ithreshold !total trajectory distribution
double precision,parameter:: tau=1d-3 !timestep
double precision,parameter:: D=1.0d0 !temperature
double precision,parameter:: drift=1.0d0 !external noneq force
double precision:: a1 !neq part of Doob force
double precision,parameter:: pi=4.0d0*atan(1.0d0)
double precision,parameter:: V0=2.0d0 !Amplitude of original potential
integer,parameter:: mintot=N !Total number of minimas
double precision, parameter:: L=2.0d0*mintot*pi !length of box
double precision, parameter:: wvnm =2.0d0*pi*mintot/L!wavenumber k of the potential V0coskx
double precision, parameter:: lambda=L/mintot
!Parameters useful in force calculation
double precision,parameter :: etavar=(2.0d0*D*tau)**0.5d0 !variance of the gaussian white noise

!data structures
double precision, dimension(N):: x0,x1,x2,F1,F0
double precision, dimension(N) :: rand1,rand2,r,theta,eta
double precision, dimension(2*M,N) :: delu
double precision :: grnd
integer :: sizerand
integer, allocatable :: putrand(:)
integer, dimension(8) :: values,valuesn !!for timestamp

!counters
integer(kind=8) :: i,j,k,p

!reinforcement stuff
double precision:: alphar=1d-4,alphapsi=1d-1,alphatheta=1d2
!double precision, dimension
integer, parameter:: M3=5
double precision, dimension(2*M3+1):: Vparams,delV
double precision, dimension(2*M):: dela
double precision:: rbar,delta,deltai,Vval,oldvval,rstep
double precision, dimension(N):: deltax
double precision, parameter:: gamma=0.99d0

!files
open(unit=20,file='position.txt')
open(unit=101,file='aparams.txt')
open(unit=102,file='Vparams.txt')
open(unit=103,file='rbar.txt')
open(unit=104,file='delta.txt')

do j=1,N
    x0(j)=(j-0.5d0)*lambda
end do

a1=drift+2.*D*kldf
Vparams=0.0
a=0.0
a(5)=1.0
rbar=0.0
dela=0.0d0
delV=0.0d0
delta=0.0d0

x2=x0

call date_and_time(VALUES=values)

call random_seed(size=sizerand)
allocate(putrand(sizerand))
putrand(:)=1d6*my_id+walkerno
call random_seed(put=putrand)

do i=1,steps

    x1=x2

    !generates random numbers for propagation
    do j=1,N
        if (mod(i,2)==1) then
            do while(.true.)
!                rand1(j)=grnd()
                call random_number(rand1(j))
                if (rand1(j) .gt. 1d-300) exit
            end do
            do while(.true.)
!                rand2(j)=grnd()
                call random_number(rand2(j))
                if (rand2(j) .gt. 1d-300) exit
            end do
            r(j)=(-2.0d0*log(rand2(j)))**0.5d0
            theta(j)=2.0d0*pi*rand1(j)
            rand1(j)=r(j)*cos(theta(j))
            rand2(j)=r(j)*sin(theta(j))
            eta(j)=rand1(j)
        else
            eta(j)=rand2(j)
        end if
    end do

    !force calculation
    call force(x1,drift,a1,N,L,F1,V0,wvnm,lambda,pi,a,M,delu)

    !original force calculation
    F0=V0*wvnm*sin(wvnm*x1)+drift

    !update position
    deltax=tau*F1+etavar *eta
    x2=x1+deltax

    !Find Vval here
    oldvval=Vval
    Vval=Vparams(2*M3+1)
    do j=1,N
        do k=1,M3
            Vval=Vval+Vparams(k)*cos(k*x2(j)*wvnm)+Vparams(k+M3)*sin(k*x2(j)*wvnm)
        end do
    end do

    !periodic boundary conditions
    do j=1,N
        x2(j)=x2(j)-L*floor(x2(j)/L)
    end do

    if (i>ithreshold .and. mod(i-ithreshold,1)==0) then
    !Do the reinforcement stuff here
    rstep=0.0d0
    do j=1,N
    rstep=rstep+kldf*deltax(j)/tau+((deltax(j)/tau-F1(j))**2-(deltax(j)/tau-F0(j))**2)/(4.0d0*D)
    end do
    deltai=Vval+(rstep-rbar)*tau-oldvval
    delta=delta+deltai
    do k=1,M
        do j=1,N
            dela(k)=dela(k)+alphatheta*deltai*etavar*eta(j)*delu(k,j)/(2.0d0*D)
            dela(k+M)=dela(k+M)+alphatheta*deltai*etavar*eta(j)*delu(k+M,j)/(2.0d0*D)
        end do
    end do
    do k=1,M3
        do j=1,N
            delV(k)=delV(k)+alphapsi*deltai*cos(k*x1(j)*wvnm)
            delV(k+M3)=delV(k+M3)+alphapsi*deltai*sin(k*x1(j)*wvnm)
        end do
    end do
    delV(2*M3+1)=delV(2*M3+1)+alphapsi*deltai
    Vparams(:)=Vparams(:)+delV(:)
    delV=0.0d0
    if (mod(i-ithreshold,int(1d0))==0) then
        a(:)=a(:)+dela(:)/1d0
        rbar=rbar+alphar*delta/(1d0*tau)
        delta=0.0d0
        dela=0.0d0
    end if

    end if
    !output
    if (mod(i-ithreshold,int(1d3))==0) then
        !print *, i-ithreshold,rbar,a(1),Vparams(1)
        write(101,'(8(2x,E15.4))')a(:)
        write(102,'(41(2x,E15.4))')Vparams(:)
        write(103,'(1(2x,E15.4))')rbar
        write(104,'(1(2x,E15.4))')deltai
        write(20,*)x2(1)
    end if
    call flush()

    if (i-ithreshold == 1d5) then
        alphar = 1d-5
    end if

end do


end program srk2
!*****************************************************************************80
subroutine force (x,drift,a1,N,L,F,V0,wvnm,lambda,pi,a,M,delu)
implicit none

integer :: i,j,k,N,M
double precision :: x(N),drift,a1,F(N),L,V0,wvnm,lambda
double precision :: pi,a(2*M),delu(2*M,N)

F=0.0d0
delu=0.0d0

do i=1,N

    F(i)=F(i)+a1
    do k=1,M
        delu(k,i)=cos(k*x(i)*wvnm)
        delu(k+M,i)=sin(k*x(i)*wvnm)
        F(i)=F(i)+a(k)*delu(k,i)+ a(k+M)*delu(k+M,i)
    end do

end do

end subroutine force
!*****************************************************************************80
!*****************************************************************************80
!************************************************************************
!*****************************************************************************80
