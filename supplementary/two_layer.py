#!/usr/bin/env python
'''
script used to generate figure 1, which compares the postseismic approximation
given by 11 to the series solution in 10
'''
from __future__ import division
import numpy as np
import ilt
import elastic
import matplotlib.pyplot as plt

year2sec = 3.155e7 # sec/year

## model parameters
mu1 = 3.2e10 # shear modulus of top layer (Pa)
mu2 = 3.2e10 # shear modulus of substrate (Pa)
eta1 = 1e18 # viscosity of top layer (Pa s)
eta2 = 1e19 # viscosity of substrate (Pa s)
tau1 = eta1/mu1 # relaxation time of top layer (s)
tau2 = eta2/mu2 # relaxation time of substrate (s)
D = 10.0 # locking depth (km)
H = 15.0 # thickness of top layer (km)
x1 = np.array([10.0,20.0,40.0,80.0]) # timeseries output locations (km)
x2 = np.linspace(0.01,80,100) # mapview output locations (km)
t1 = np.linspace(0.0,2.0,100) # timeseries output times (years)
t2 = np.array([0.0,0.5,1.0,1.5]) # mapview output times (years)

## convert eta_i and tau_i to years
tau1 = tau1/year2sec # year
tau2 = tau2/year2sec # year
eta1 = eta1/year2sec # Pa year
eta2 = eta2/year2sec # Pa year

## nondimensionalize
tau_min = min([tau1,tau2])
x1_ = x1/D # output locations in terms of locking depths
x2_ = x2/D 
t1_ = t1/tau_min # output times in terms of min(tau_i)
t2_ = t2/tau_min

## ILT parameters
s_min = 1e6
nmax = 12

## Laplace transform of viscoelastic solution
def uhat(s,x):
  mu1hat = s/((s/mu1) + (1/eta1)) # effective top layer shear modulus
  mu2hat = s/((s/mu2) + (1/eta2)) # effective substrate shear modulus
  bhat = 1/s # laplace transform of unit slip
  # call the function which computes the elastic solution
  return elastic.two_layer(x,D,bhat,mu1hat,mu2hat,H,nmax)

# compute the viscoelastic solution in the time domain for mapview
# locations
u = ilt.ILT(uhat,nmax,f_args=(x2,),s_min=s_min)

plt.figure(1)
plt.plot(x2_,u(t2),'b',lw=2)

# compute the viscoelastic solution in the time domain for timeseries
# locations
u = ilt.ILT(uhat,nmax,f_args=(x1,),s_min=s_min)
plt.figure(2)
for i in range(len(x1)):
  plt.plot(t1_,u(t1)[i],'b',lw=2)
plt.xlim((0,2))

# compute the viscoelastic two term approximation for mapview
# locations
u = ilt.ILT(uhat,2,f_args=(x2,),s_min=s_min)
plt.figure(1)
plt.plot(x2_,u(t2),'k--',lw=2)
plt.xlabel('x/D',fontsize=16)
plt.ylabel('u/slip',fontsize=16)
plt.text(x2_[50],u(t2)[50][0]-0.04,'t=0.0 tau2',fontsize=16)
plt.text(x2_[50],u(t2)[50][-1]+0.005,'t=1.5 tau2',fontsize=16)

# compute the viscoelastic two term approximation for timeseries
# locations
u = ilt.ILT(uhat,2,f_args=(x1,),s_min=s_min)
plt.figure(2)
for i in range(len(x1_)):
  plt.plot(t1_,u(t1)[i],'k--',lw=2)
  plt.text(0.2,u(t1)[i][0]-0.01,'x=%s D' %x1_[i],fontsize=16)
plt.xlim((0,2))

plt.xlabel('t/tau2',fontsize=16)
plt.ylabel('x/slip',fontsize=16)

plt.show()

