# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:00:18 2013

@author: dgevans
"""
import bellman
import numpy as np
import matplotlib.pyplot as plt
#this code solves and plots the solution for several levels of theta


Vf0 = lambda x: -0.5 * x**2

#theta = 0.5
bellman.theta = 0.5
Vf05 = bellman.solve_bellamn(Vf0)

#theta = 1.
bellman.theta = 1.
Vf1 = bellman.solve_bellamn(Vf0)

#theta = 3.
bellman.theta = 3.
Vf3 = bellman.solve_bellamn(Vf0)

#plot policy
xgrid = np.linspace(-15.,15.,200)
plt.plot(xgrid,bellman.upolicy(Vf05,xgrid))
plt.plot(xgrid,bellman.upolicy(Vf1,xgrid))
plt.plot(xgrid,bellman.upolicy(Vf3,xgrid))

plt.xlabel('x')
plt.ylabel('u')
plt.legend(('theta = 0.5','theta = 1.','theta=3.'))

plt.savefig('Images/fig1.eps')
plt.close()

bellman.store_parameters(Vf0)
Vf0.theta = 0.5
plt.plot(xgrid,bellman.upolicy(Vf0,xgrid))
Vf0.theta = 1.
plt.plot(xgrid,bellman.upolicy(Vf0,xgrid))
Vf0.theta = 3.
plt.plot(xgrid,bellman.upolicy(Vf0,xgrid))

plt.xlabel('x')
plt.ylabel('u')
plt.legend(('theta = 0.5','theta = 1.','theta=3.'))
plt.savefig('Images/fig2.eps')