import numpy as np
import matplotlib.pyplot as plt

rbar=np.loadtxt('rbar.txt')
plt.plot(rbar)
plt.ylabel(r'$\bar{r}$')
plt.show()

#delta=np.loadtxt('delta.txt')
#plt.plot(delta)
#plt.ylabel(r'$\delta$')
#plt.show()

aparams=np.loadtxt('aparams.txt')
plt.plot(aparams)
plt.ylabel(r'aparams')
plt.show()

Vparams=np.loadtxt('Vparams.txt')
plt.plot(Vparams)
plt.ylabel(r'Vparams')
plt.show()
print(Vparams[-1,:])

pos=np.loadtxt('position.txt')
plt.plot(pos)
plt.ylabel(r'Position')
plt.show()
