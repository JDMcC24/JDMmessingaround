import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time 
import sympy as sp


# Plotting multiple generations 
# f_n(x) = \int K(x,y) D(y) f_{n-1}(y) dy. In simplest cases
# K(x,y) = k(x-y) where k is a probability density function defined
# on the shared domain with f_n. D is an continuous positive function
# denoting the intrinstic repoductive rate of the population at 
# location x. 

startt = time.time()
L = 1 #Defining domain of population

#Dispersal Kernal 
def DispersalKernal(x):
    return 1/(2*L) if x >= -L and x <= L else 0

#Initial population Density function
def initpop(x): 
    return L - (2**(-1))*abs(x+L) if x> -L and x <= L else 0

#Local Intrinsics Reproduction Function
def locrep(x):
    return 1.05 if x>=-L and x<=L else 0



#Plotting Initial population density
dom = np.linspace(-5, 5, 1000) #Getting window larger than domain for contrast
K = [DispersalKernal(x) for x in dom]
f0 = [initpop(x) for x in dom]
#plt.plot(dom,K, label='Dispersal Kernel')
plt.plot(dom,f0, label='Initial Population Density')
#Definining new population desnity values after reproduction
#f0 = [locrep(x)* initpop(x) for x in dom]
fn = np.convolve(f0,K, 'same')
fn = (10) * fn/ len(fn) #Normalizing the convolution
dom2 = np.linspace(-5,5, len(fn)) #Plotting the generation
plt.plot(dom2,fn, label = 'Generation 2' )
for i in range (1,3):
    #fn = f0 = [locrep(dom2[x])* fn[x] for x in range(len(dom2))]
    fn = np.convolve(fn,K, 'same')
    fn = (10) * fn/ len(fn)
    dom2 = np.linspace(-5,5, len(fn))
    plt.plot(dom2,fn, label=  f'Generation {i+2}')
plt.legend()
plt.title('Uniform Dispersal')
plt.suptitle('Population Density over Generations')
print(time.time() - startt)
plt.show()

s, t = sp.symbols('s t')
print(sp.integrate((s - 0.5*(t+s))*(2*(s))**(-1), (t,-s,s)))