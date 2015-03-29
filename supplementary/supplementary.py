
#!/usr/bin/env python

import sympy as sp
from sympy.solvers import solve
import numpy as np
import matplotlib.pyplot as plt

def nth_derivative_at_zero(uhat,s,n):
  '''                 
  returns the the nth derivative of u(x) evaluated at zero given the laplace   
  transform of u(x)                        
  PARAMETERS                                 
  ----------                                             
    uhat: laplace transform of u. This is a symbolic function of s.     
    s: the laplace domain variable                           
    n: the derivative order                            
                                                                 
  RETURNS                                           
  -------                                                      
    nth_derivative: symbolic expression for the nth derivative of u evaluated
                    at zero                          
  '''
  assert n >= 0
  if n == 0:
    expr = s*uhat
    zeroth_derivative = expr.limit(s,np.inf)
    return zeroth_derivative
  elif n > 0:
    expr = s**(n+1)*uhat
    # note that m goes up to n-1           
    for m in range(0,n):
      expr -= s**(m+1)*nth_derivative_at_zero(uhat,s,n-m-1)
    nth_derivative = expr.limit(s,np.inf)
    return nth_derivative

def inverse_laplace_transform_series_expansion(uhat,s,t,N):
  '''                                                        
  returns the Taylor series expansion of u(x) about zero given its laplace 
  transform                                    
                                                              
  PARAMETERS                                      
  ----------                                         
    uhat: symbolic expression for the laplace transform of u(x)     
    s: laplace domain variable                          
    N: order of the Taylor series expansion     
                                                   
  RETURNS                                   
  -------                                        
    series: symbolic expression for the series expansion of the u(x) 
          about x=0        
  '''
  series = 0
  # note that n goes up to N     
  for n in range(N+1):
    # sum each term in the taylor series expansion   
    series += (nth_derivative_at_zero(uhat,s,n)*t**n)/sp.factorial(n)
  return series

x,s = sp.symbols('x s')

u = 3*x**2 + 2*x + 1 # test function              
uhat = sp.laplace_transform(u,x,s)[0] # test function laplace transform                   
u_expansion = inverse_laplace_transform_series_expansion(uhat,s,x,2)
assert sp.simplify(u - u_expansion) == 0

'''

Maxwell viscoelastic 2D two layered earthquake model 
  I am demonstrating here that the surface deformation resulting from viscous  
  relaxation following an earthquake in a layered halfspace is, to first order, 
  a linear function of the viscosities in each layer.      

Variables used
  x: distance from fault strike         
  t: time                    
  D: locking depth of the fault         
  H: thickness of the top layer    
  mu1,mu2: shear modulus of the first and second layer  
  eta1,eta2: viscosity of the first and second layer

'''
x,t,D,H,theta = sp.symbols('x t D H theta')
mu1,mu2 = sp.symbols('mu1 mu2')
eta1,eta2 = sp.symbols('eta1 eta2')
b = sp.Function('b')(t)
n = sp.symbols('n') # dummy summation variable   
nmax = 3 # The number of terms used to approximate the infinite series

'''
The surface displacements for a layered viscous halfspace can be found by using
the displacements for a layered elastic halfspace and the correspondence     
principle. The surface displacements for a layered elastic halfspace are:
'''
gamma = (mu1-mu2)/(mu1+mu2)
W = sp.Function('W')(n)
# uncomment for explicit W   
#W = 1/sp.pi*(sp.atan((D + 2*n*H)/x) + sp.atan((D - 2*n*H)/x))         
u_elastic = b*(sp.Rational(1,2)*W.subs(n,0) + sp.summation(gamma**n*W,(n,1,nmax)))

'''
The correpondence principle says that the Laplace transform of displacements 
in an elastic medium is the same form as the displacements in a viscoelastic
medium. The laplace transform of displacements for an elastic medium are:
'''
uhat_elastic = sp.laplace_transform(u_elastic,t,s)

'''
Replacing the shear modulii with the equivalent maxwell viscoelastic shear
modulii in the laplace domain gives us the laplace transform of displacements
in the viscoelastic model 
'''
mu1hat = s/((s/mu1) + (1/eta1)) # equivalent shear modulii
mu2hat = s/((s/mu2) + (1/eta2))
uhat_viscoelastic = uhat_elastic.subs(((mu1,mu1hat),(mu2,mu2hat)))

'''
for the sake of simplicity, although this does not reduce the generality of our
conclusions, we assume that the viscoelastic model has a homogeneous shear modulus
'''
uhat_viscoelastic = uhat_viscoelastic.subs(mu2,mu1)

'''
We are interested in the viscous deformation and so we remove the elastic 
component of deformation.  Doing so also removes a singularity in the 
displacement field at the time of the earthquake making subsequent computations 
easier.                 
'''
uhat_viscous = uhat_viscoelastic - uhat_elastic.subs(mu2,mu1)

'''
Using the initial value theorem, the first term in the Taylor series expansion
of the viscous deformation is b(t) convolved with:
'''
bhat = sp.laplace_transform(b,t,s)
u_viscous_term0 = inverse_laplace_transform_series_expansion(uhat_viscous/bhat,s,t,0)
sp.pprint(sp.expand(u_viscous_term0))

'''
We can find the next few  terms of the Taylor series expansion of viscous 
deformation using an extension of the initial value theorem.  The viscous 
deformation is b(t) convolved with:
'''
u_viscous_term0to2 = inverse_laplace_transform_series_expansion(uhat_viscous/bhat,s,t,2)
sp.pprint(sp.collect(sp.expand(u_viscous_term0to2),t))

'''
An approximation for early postseismic deformation can then be found by 
combining the elastic deformation with the first term in the Taylor series of the 
viscous component of deformation.
'''
u_elastic = u_elastic.subs(mu2,mu1)
u_approx = u_elastic + sp.integrate(b.subs(t,theta)*u_viscous_term0,(theta,0,t))
sp.pprint(u_approx)

'''
Maxwell viscoelastic 2D three layered earthquake model
  This procedure can be repeated for a three layered viscoelastic earthquake 
  model.  We start with the solution for the three layered elastic solution from 
  Chinnery and Jovanovich 1972. WE ADOPT THE NOTATION FROM CHINNERY AND
  JOVANOVICH 1972 which is described below
                     
variables used:                    
  mu3,mu2,mu1: shear modulus for the top, middle, and bottom layer (mu4 from 
               Chinnery and Jovanovich is assumed 0)    
  eta3,eta2,eta1: viscosity for the top, middle, and bottom layer 
  h3,h2: thichness of the top and middle layer       
  p: depth to the bottom of the fault. (down is positive)
'''

l_max = 2
m_max = 2
n_max = 2

h2,h3,p,y = sp.symbols('h2 h3 p y')
mu1,mu2,mu3 = sp.symbols('mu1 mu2 mu3')
eta1,eta2,eta3 = sp.symbols('eta1 eta2 eta3')
M,N = sp.symbols('M N')

a1    = (mu2 - mu1) / (mu2 + mu1)
a2    = (mu3 - mu2) / (mu3 + mu2)
a3    = -1
b1    = 2*mu1 / (mu1 + mu2)
b2    = 2*mu2 / (mu2 + mu3)
b3    = 2
d1    = 2*mu2 / (mu1 + mu2)
d2    = 2*mu3 / (mu2 + mu3)
W = sp.Function('W')(M,N)
# uncomment for explicit W                                                            
#W = 1/pi*(sp.atan((-2*M*h2 - N*h3 + p)/y) + a3*sp.atan((-2*M*h2 - N*h3 - p)/y))

def P(l,m,n):
  N = sp.factorial(n+1) * sp.factorial(l+m-1)
  D = sp.factorial(l) * sp.factorial(n) * sp.factorial(l-1) * sp.factorial(m)
  return N / D

def Q(l,m,n):
  N = sp.factorial(n+l) * sp.factorial(l+m)
  D = sp.factorial(l) * sp.factorial(n) * sp.factorial(l) * sp.factorial(m)
  return N / D

u = -b3*sp.Rational(-1,2)*W.subs(((N,0),(M,0)))
for n in range(n_max):
  coeff = a2*b3*(-a2*a3)**n
  u += coeff*W.subs(((M,0),(N,n+1)))

for l in range(1,l_max):
  for m in range(m_max):
    for n in range(n_max):
      coeff = a2*b3*(-a1*a2)**m*(-a2*a3)**n*(-a2*a3*d2*b2)**l*P(l,m,n)
      u += coeff*W.subs(((M,l+m),(N,l+n+1)))

for l in range(l_max):
  for m in range(m_max):
    for n in range(n_max):
      coeff = a1*d2*b2*b3*(-a1*a3)**m*(-a2*a3)**n*(-a1*a3*d2*b2)**l*Q(l,m,n)
      u += coeff*W.subs(((M,l+m+1),(N,l+n+1)))

u_elastic = sp.Rational(1,2)*b*u

'''
Assuming a homogeneous shear modulus, mu1, the Laplace transform of the 
viscous component of deformation is: 
'''
uhat_elastic = sp.laplace_transform(u_elastic,t,s)
mu1hat = s/((s/mu1) + (1/eta1))
mu2hat = s/((s/mu1) + (1/eta2))
mu3hat = s/((s/mu1) + (1/eta3))
uhat_viscoelastic = uhat_elastic.subs(((mu1,mu1hat),(mu2,mu2hat),(mu3,mu3hat)))
uhat_viscous = uhat_viscoelastic - uhat_elastic.subs(((mu2,mu1),(mu3,mu1)))

'''
The first term in the Taylor series expansion of the viscous component of             
deformation is b(t) convolved with (this will take a few minutes):
'''
u_viscous_term0 = inverse_laplace_transform_series_expansion(uhat_viscous/bhat,s,t,0)
u_viscous_term0 = sp.expand(u_viscous_term0)
sp.pprint(u_viscous_term0)

'''
An approximation for early postseismic deformation can then be found by                 
combining the elastic deformation with the first term in the Taylor series of the  
viscous component.
'''
theta = sp.symbols('theta')
u_elastic = u_elastic.subs(((mu2,mu1),(mu3,mu1)))
u_approx = u_elastic + sp.integrate(b.subs(t,theta)*u_viscous_term0,(theta,0,t))
sp.pprint(u_approx)






