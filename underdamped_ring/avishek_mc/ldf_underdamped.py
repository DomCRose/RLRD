from pylab import *
from scipy import linalg as LA

# Underdamped diffusion on a ring
# Height the periodic potential
vo=2.0
# temperature
temp=1.0
# Drfit force
f=1.
# Mass
mass=1.
# friction
gamma=1.

# Position Basis size 2*Mx+1
Mx=20
# Velocity basis size Mv+1
Mv=30
# Full tilted operator
Ltilt=zeros([(2*Mx+1)*(Mv+1),(2*Mx+1)*(Mv+1)])*0j
# Position subspace tilted operators
xtilt1=zeros([2*Mx+1,2*Mx+1])*0j
xtilt2=zeros([2*Mx+1,2*Mx+1])*0j
# Lambda is the tilt parameter
lams=arange(-1.5,0.6,0.1)
print(lams)
# Container for the LDF
psi=zeros(len(lams))
# Counter
j=0
for i in range(0,2*Mx+1,1):
    n=i-Mx
    xtilt1[i,i]=(1.j*n*temp-f)
    xtilt2[i,i]=1.j*n*(temp/mass)**0.5
    if i<2*Mx:
        xtilt1[i,i+1]=1.j*vo/2.
    if i>0:
        xtilt1[i,i-1]=-1.j*vo/2.
xtilt1=xtilt1/(temp*mass)**0.5
for lam in lams:
    #Construct Ltilt
    for p in range(0,Mv+1):
        Ltilt[p*(2*Mx+1):(p+1)*(2*Mx+1),p*(2*Mx+1):(p+1)*(2*Mx+1)]=-p*gamma/mass *identity(2*Mx+1)
        if p<Mv:
            Ltilt[p*(2*Mx+1):(p+1)*(2*Mx+1),(p+1)*(2*Mx+1):(p+2)*(2*Mx+1)]=-(p+1)**0.5 *(xtilt2-lam*(temp/mass)**0.5 *identity(2*Mx+1))
        if p>0:
            Ltilt[p*(2*Mx+1):(p+1)*(2*Mx+1),(p-1)*(2*Mx+1):p*(2*Mx+1)]=-(p)**0.5 *(xtilt1-lam*(temp/mass)**0.5 *identity(2*Mx+1))
    #Diagonalize Ltilt to find left eigenvector and eigenvalue
    psis,v=LA.eig(Ltilt,left=True,right=False)
    psi[j] = max([i for i in psis if abs(i.imag) < 1e-8]).real
    index=where(abs(psis-psi[j]) < 1e-8)[0]
    #print(where(abs(psis-psi[-1]) < 1e-8))
    savetxt("eig_"+str(Mx)+"_"+str(Mv)+"_"+str(lam)+".txt",v[:,index].view(float))
    #print(shape(v[:,index]))
    j=j+1
    
plot(lams,psi,'-',lw=3.)
savetxt("underdamped_psi_exact.txt",psi)

xlabel(r'$\lambda$',size=20)
ylabel(r'$\psi(\lambda$)',size=20)
#xticks(size=18)
#yticks(size=18)
#axis([-10,5,-1,5])
tight_layout()
show()

