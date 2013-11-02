# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:25:12 2013

@author: dgevans
"""
#load necessary modules
from numpy import *
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from itertools import izip

#set parameters
mu = 1. #mean of beta
sigma = 0.2 #standard deviationof sigma
delta = 0.1 #exp(-delta) is discount factor
alpha = 0.6 #coefficient on W_{t+1}
theta = 1. #risk sensitivity parameter
kappa = 0.8 #coefficient on $x$

#interpolation region on xprime
min_xprime = -15.
max_xprime = 15.
xprimegrid = linspace(min_xprime,max_xprime,25)

#Compute Gaussian quadrature nodes
#nodes for $W_{t+1}
wvec,w_weights = hermgauss(15)
wvec *= sqrt(2) #sqrt(2) because nodes are for weighting function e^(-x^2)
w_weights /= sum(w_weights)

#Nodes for $\beta$ depends on mu an sigma, so need to set those with function
#set_mu_sigma
def set_mu_sigma(_mu,_sigma):
    global mu,sigma,betavec,beta_weights
    mu = _mu
    sigma = _sigma
    betavec,beta_weights = hermgauss(15)
    betavec= mu + sigma*betavec*sqrt(2)
    beta_weights /= sum(beta_weights)    
set_mu_sigma(mu,sigma)
    
class F:
    '''
    First Bellman Map takes a function Vf and returns function object corresponding 
    to E[Vf(alpha*W_{t+1}+beta*u+kappa*x) | x,beta,u]
    '''
    def __init__(self,Vf):
        '''
        Inits the class with the value functio Vf
        '''
        self.Vf = Vf
        
    def __call__(self,x,beta,u):
        '''
        Evaluates the expectation of the value function conditional on bet and u
        '''
        xprime = alpha*wvec+beta*u+kappa*x
            
        return w_weights.dot(self.Vf(xprime))
        
    
class T2:
    '''
    Performs the T^2 bellman map, returns the function object [T^2\tilde V](x,u)
    '''
    def __init__(self,Vtilde):
        '''
        Inits with value function Vtilde
        '''
        self.Vtilde = Vtilde
        
    def __call__(self,x,u):
        '''
        Perfoms the T2 map by taking expectation over beta
        '''
        Vtilde = zeros(betavec.shape)
        for i,beta in enumerate(betavec):
            Vtilde[i] = self.Vtilde(x,beta,u)
        log_beta_weights = log(beta_weights)
        temp = exp(log_beta_weights  - exp(-delta)/theta*Vtilde )
        return -theta*log( sum(temp)  )
        
class T:
    '''
    Performs the final bellman map by maximizing over u.
    '''
    def __init__(self,T2V):
        '''
        Inits with the value function T2V
        '''
        self.T2V = T2V
        
    def __call__(self,x):
        '''
        Minimizes the objective with respect to u
        '''
        ubar = get_ubar() #we now no bounds on u so we can use them in maximization
        res = minimize_scalar(lambda u: -self.T2V(x,u),bounds=(-ubar,ubar),method='bounded')
        return -0.5*x**2 - res.fun
        
def get_ubar():
    '''
    Computes upper bound for u
    '''
    return sqrt(theta/(sigma**2 * exp(-delta)))
        
def store_parameters(Vf):
    '''
    Stores the parameters in the value function
    '''
    Vf.mu = mu
    Vf.sigma = sigma
    Vf.delta = delta
    Vf.alpha = alpha
    Vf.theta = theta
    Vf.kappa = kappa
    
def load_parameters(Vf):
    '''
    Loads parameters from the Value function
    '''
    global delta,alpha,theta,kappa
    delta = Vf.delta
    alpha = Vf.alpha
    theta = Vf.theta
    kappa = Vf.kappa
    set_mu_sigma(Vf.mu,Vf.sigma)

def iterate_bellman(Vf):
    '''
    Iterates the bellman equation given a value function Vf
    '''
    Vtilde = F(Vf)#perform first bellman map
    T2V = T2(Vtilde)#second bellman map
    Vnew = T(T2V)#third bellman map to get new value function
    
    #Now approximate value function
    Vs = hstack(map(Vnew,xprimegrid)) #apply Vnew to each element of grid
    Vfnew = ValueFunction(xprimegrid,Vs) #interpolate function 
    store_parameters(Vfnew) #store parameters in vale function
    return Vfnew
    
def solve_bellamn(Vf0):
    '''
    Iterates until convergence
    '''
    Vs = Vf0(xprimegrid)#store pevious values
    Vf = Vf0
    diff = 1.
    while diff > 1e-5:
        Vfnew = iterate_bellman(Vf)
        diff = amax(abs((Vs-Vfnew.y)/Vfnew.y))
        Vs = Vfnew.y
        print diff
        Vf = Vfnew
    return Vf
    
def upolicy(Vf,xgrid):
    '''
    Compute the u policy over xgrid
    '''
    load_parameters(Vf) #load parameters associated with value function
    Vtilde = F(Vf) #apply first bellman map
    T2V = T2(Vtilde) #apply second bellman map
    def f(x): #finds optimal $u$
        ubar = get_ubar()
        return minimize_scalar(lambda u: -T2V(x,u),bounds=(-ubar,ubar),method='bounded').x
    return hstack(map(f,xgrid)) #compute optimal $u$ for each x in xgrid
    
class ValueFunction(object):
    '''
    Value function class, handles the interpolation
    '''
    def __init__(self,x,Vs):
        '''
        Initializes 
        '''
        self.x = x #store x
        self.y = Vs #store Vs in y
        self.f = interp1d(x,Vs,bounds_error=False,kind='cubic') #interolate
        
        #compute quadratic for evaluating x outside [xprime_min,xprime_max]
        X = vstack((ones(len(x)),x,x**2)).T
        self.c = linalg.lstsq(X,Vs)[0] #find best quadratic approximation of value function points
        
        
    def __call__(self,xgrid):
        '''
        Evaluate value function at points
        '''
        Vs = self.f(xgrid) #evalute inerpolation at all points on xgrid
        for i in range(len(xgrid)):
            if isnan(Vs[i]): #if nan implies outside bounds so use quadratic continuation
                Vs[i] = float(self.c.dot([1,xgrid[i],xgrid[i]**2]))
                
        return Vs
        
def T2V_1period(x,u):
    '''
    Returns the T2V value function or the 1 period problem with alpha = 0
    '''
    return 0.5*(mu**2*theta/sigma**2 -  kappa**2*x**2*exp(-delta)
    - (theta*mu+u*kappa*x*exp(-delta)*sigma**2)**2/( sigma**2*(theta-u**2*sigma**2*exp(-delta)))
    - log(theta/(theta-u**2*sigma**2*exp(-delta))) )