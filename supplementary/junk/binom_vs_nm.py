#!/usr/bin/env python
import numpy as np
from scipy.special import binom
from scipy.special import factorial
import matplotlib.pyplot as plt
import mpmath

# work dps = 3/10 * max(N) 

workdps = 100

x = np.linspace(0.01,10,20)
t = np.array(mpmath.linspace(0,80,20))
H = 2.0
D = 1.0
tau = 1.0

def W(n):
  return 1.0/np.pi*(np.arctan((D+2*n*H)/x) + np.arctan((D-2*n*H)/x))

def u_binom(N):
  out = np.zeros((len(x),len(t)))
  out = out + 0.5*W(0)[...,None]*np.ones(len(t))
  with mpmath.workdps(workdps):
    for n in range(1,N):
      coeff = np.zeros(len(x))    
      for m in range(n):     
        coeff = coeff + (-1)**((n-1)-m)*mpmath.binomial(n-1,m)*W(m+1)
      out = out + coeff[...,None]*((t/(2*tau))**n)/mpmath.factorial(n)       
  return out

plt.plot(x,u_binom(200))
plt.show()
