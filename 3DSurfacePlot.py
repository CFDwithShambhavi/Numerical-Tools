#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:45:28 2020

@author: snandan
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

data = lambda x, y: np.sin(np.sqrt(np.square(x) + np.square(y)))

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)

X, Y = np.meshgrid(x, y)
Z = data(X, Y)

ax = plt.axes(projection='3d')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)
    
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='gist_earth_r', edgecolor='none')

#ax.view_init(60, 35)