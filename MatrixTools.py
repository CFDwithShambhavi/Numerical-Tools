#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:59:11 2020

@author: snandan
"""

import numpy as np

def i_j_to_k (index, columns): 
    return index[0] * columns + index[1]

def R (index): 
    return (index[0], index[1] + 1)

def L (index): 
    return (index[0], index[1] - 1)

def T (index): 
    return (index[0] - 1, index[1])

def B (index): 
    return (index[0] + 1, index[1])

def assemble_a_to_A(rows, columns, aPs, aEs, aWs, aNs, aSs):
    
    A = np.zeros((rows * columns, rows *  columns))
    
    row = 0
    for index in np.ndindex(rows, columns):
        
        y = np.zeros(rows * columns)
        
        k = i_j_to_k(index, columns)
        y[k] = aPs[index]
        
        if index == (0, 0): # top left corner cell
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
        
        elif index == (rows - 1, 0): # bottom left corner cell
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
        elif index == (0, columns - 1): # top right corner cell
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
            
        elif index == (rows - 1, columns - 1): # bottom right corner cell
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
        elif index[1] == 0: # column 0 cells excluding top left and bottom left corner
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
            
        elif index[0] == 0: # row 0 cells excluding top left and top right corner
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
            
        elif index[1] == columns - 1: # column columns - 1 cells excluding top and bottom corner
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
            
        elif index[0] == rows - 1: # row rows - 1 cells excluding top and bottom corner
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
        else: # all interior cells
            W_index = L(index)
            k = i_j_to_k(W_index, columns)
            y[k] = -aWs[index]
            
            E_index = R(index)
            k = i_j_to_k(E_index, columns)
            y[k] = -aEs[index]
            
            N_index = T(index)
            k = i_j_to_k(N_index, columns)
            y[k] = -aNs[index]
            
            S_index = B(index)
            k = i_j_to_k(S_index, columns)
            y[k] = -aSs[index]
        
        A[row, :] = y[:] 
        
        row += 1
    
    return A