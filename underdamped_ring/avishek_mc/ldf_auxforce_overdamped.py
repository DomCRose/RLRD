from pylab import *
from numpy import linalg as LA

# Overdamped diffusion on a ring
# Height the periodic potential
vo=2.0
# Temperature
sig=2.0
# Drift force
f=1.0

# Basis size 2*M+1
M=30
# Tilted operator
Ltilt=zeros([2*M+1,2*M+1])*0j
# Lambda is the tilt parameter
lams=arange(-1.5,0.6,0.1)
# Container for the LDF
psi=zeros(len(lams))
# Counter
j=0
for lam in lams:
    for i in range(0,2*M+1,1):
        n=i-M
        Ltilt[i,i]=1.j*n*(f+sig*lam)-sig*n**2/2.+lam*f+sig*lam**2/2.
        if i<2*M:
            Ltilt[i,i+1]=vo*(1.j*lam-1-n)/2.
        if i>0:
            Ltilt[i,i-1]=vo*(-1.j*lam-1+n)/2.
    psis,v=LA.eig(Ltilt)
    #print psis, max(psis.real), max(psis)
    psi[j] = max([i for i in psis if abs(i.imag) < 1e-8]).real
    j=j+1

plot(lams,psi,'-',lw=3.)
savetxt("overdamped_psi_exact.txt",psi)

xlabel(r'$\lambda$',size=20)
ylabel(r'$\psi(\lambda$)',size=20)
#xticks(size=18)
#yticks(size=18)
#axis([-10,5,-1,5])
tight_layout()
show()

#calculating force from the eigenvector
#f_handle=open("auxforce_5.0.txt",'w')
#index=where(abs(psis-psi[-1])<1e-8)[0]
#eigen=v[:,index]
#xs=np.linspace(0.,2.*pi,50)
#for x in xs:
#    sx=0.
#    snx=0.
#    for i in range(2*M+1):
#        n=i-M
#        sx+=eigen[i]*exp(1.j*n*x)
#        snx+=n*eigen[i]*exp(1.j*n*x)*1.j
#    f_handle.write("{} {} \n".format(x,np.real(snx/sx)[0]))

