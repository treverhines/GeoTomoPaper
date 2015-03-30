#!/usr/bin/env python
import sympy as sm
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
    series += (nth_derivative_at_zero(uhat,s,n)*t**n)/sm.factorial(n)
  return series

x,t,s,D,H = sm.symbols('x t s D H')
mu1,mu2 = sm.symbols('mu:2')
eta2 = sm.symbols('eta2')
nmax = 5
n = sm.symbols('n') # dummy summation variable                                                                       
b = sm.Function('b')(t)
b = 1

gamma = (mu1-mu2)/(mu1+mu2)
#W = 1/sm.pi*(sm.atan((D + 2*n*H)/x) + sm.atan((D - 2*n*H)/x))
W = sm.Function('W')(n)
u_elastic = b*(sm.Rational(1,2)*W.subs(n,0) + sm.summation(gamma**n*W,(n,1,nmax)))
uhat_elastic = sm.laplace_transform(u_elastic,t,s)[0]

mu2hat = s/((s/mu2) + (1/eta2))
uhat = uhat_elastic.subs(mu2,mu2hat)
uhat = uhat.subs(mu1,mu2)
uv_1 = nth_derivative_at_zero(uhat,s,4).expand()
sm.pprint(uv_1)

'''
#bhat = sm.laplace_transform(b,t,s)[0]

uhat = uhat.subs(mu2,mu1)

u_final = (s*uhat).limit(s,0)

u_viscous_2 = inverse_laplace_transform_series_expansion(uhat,s,t,nmax)

#sm.pprint(u_viscous)

#u_elastic = x + t
u_final_numerical = sm.lambdify((x,D,H,mu1,eta1,eta2),u_final,"numpy")
u_elastic_numerical = sm.lambdify((x,D,H,mu1,mu2),u_elastic,"numpy")
u_viscous_1_numerical = sm.lambdify((x,t,D,H,mu1,eta1,eta2),u_viscous_1,"numpy")
u_viscous_2_numerical = sm.lambdify((x,t,D,H,mu1,eta1,eta2),u_viscous_2,"numpy")

xnum = np.linspace(-100,100,500)

plt.figure(1)
plt.plot(xnum,u_viscous_2_numerical(xnum,0.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,5.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,10.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,15.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'b',lw=2)

plt.plot(xnum,u_final_numerical(xnum,10.0,20.0,3.2e10,1e20,1e19),'r',lw=2)

plt.plot(xnum,u_viscous_1_numerical(xnum,0.0,10.0,20.0,3.2e10,1e20,1e19),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,5.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,10.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,15.0*3.156e7,10.0,20.0,3.2e10,1e20,1e19),'k--',lw=2)

plt.xlabel('$\mathrm{distance\ from\ fault (km)}$',fontsize=16)
plt.ylabel('$\mathrm{displacement (meters)}$',fontsize=16)
plt.xlim(0,100)
plt.ylim(0,2.75)

plt.figure(2)

xnum = 40.0
t = 3.1567e7*np.linspace(0.0,15.0,100)
plt.plot(t/3.1567e7,u_viscous_2_numerical(xnum,t,10.0,20.0,3.2e10,1e20,1e19),'b',lw=2)

plt.plot(t/3.1567e7,u_viscous_1_numerical(xnum,t,10.0,20.0,3.2e10,1e20,1e19),'k--',lw=2)

plt.xlim((0,15))
plt.ylim((0.3,0.7))
'''

'''
plt.plot(xnum,u_viscous_2_numerical(xnum,0.0*3.156e7,1,2,3.2e10,1e19,1e20),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,5.0*3.156e7,1,2,3.2e10,1e19,1e20),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,10.0*3.156e7,1,2,3.2e10,1e19,1e20),'b',lw=2)
plt.plot(xnum,u_viscous_2_numerical(xnum,15.0*3.156e7,1,2,3.2e10,1e19,1e20),'b',lw=2)

plt.plot(xnum,u_viscous_1_numerical(xnum,0.0,1,2,3.2e10,1e19,1e20),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,5.0*3.156e7,1,2,3.2e10,1e19,1e20),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,10.0*3.156e7,1,2,3.2e10,1e19,1e20),'k--',lw=2)
plt.plot(xnum,u_viscous_1_numerical(xnum,15.0*3.156e7,1,2,3.2e10,1e19,1e20),'k--',lw=2)
'''

plt.show()

