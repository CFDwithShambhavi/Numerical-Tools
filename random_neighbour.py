#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:19:12 2020

@author: snandan
"""
import numpy as np

def neighbour(index,ny,nx):
    n_index = []
    if index[0]==0 and index[1]==0:
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0]+1,index[1]+1))
    elif index[0]==ny-1 and index[1]==nx-1:
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0]-1,index[1]-1))
    elif index[0]==0 and index[1]==nx-1:
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0]+1,index[1]-1))
    elif index[0]==ny-1 and index[1]==0:
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0]-1,index[1]+1))
    elif index[0]==0:
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0]+1,index[1]-1))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0]+1,index[1]+1))
    elif index[0]==ny-1:
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0]-1,index[1]-1))
        n_index.append((index[0]-1,index[1]+1))
    elif index[1]==nx-1:
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0]-1,index[1]-1))
        n_index.append((index[0]+1,index[1]-1))
    elif index[1]==0:
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0]-1,index[1]+1))
        n_index.append((index[0]+1,index[1]+1))
    else:
        n_index.append((index[0],index[1]-1))
        n_index.append((index[0],index[1]+1))
        n_index.append((index[0]-1,index[1]))
        n_index.append((index[0]+1,index[1]))
        n_index.append((index[0]-1,index[1]-1))
        n_index.append((index[0]+1,index[1]+1))
        n_index.append((index[0]+1,index[1]-1))
        n_index.append((index[0]-1,index[1]+1))
        
    return n_index[np.random.randint(0,len(n_index)-1)]