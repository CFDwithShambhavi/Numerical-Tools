#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:22:43 2020

@author: snandan
"""

import numpy as np
import CSV_FileReader_Writer as csv
import MatrixTools as mt
import os


def assemble_A_and_S(K,Cp,rho,Tc_old,dt,ny,nx,dx,dy,left,top,right,bottom):
    
    Alpha = K/(rho*Cp)
    
    # Scalar variable datastructure:
    ac = np.zeros((ny,nx))
    al = np.zeros((ny,nx))
    ar = np.zeros((ny,nx))
    at = np.zeros((ny,nx))
    ab = np.zeros((ny,nx))
    source = np.zeros((ny,nx))
    jmin = 1
    jmax = ny-2
    for j in np.arange(jmin, jmax+1):
        if j==jmin: # top row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==1: # left corner node:
                    
                    if top[0] == 'D' and left[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[i,j] = Tc_old[j,i]/dt + left[1]*Alpha/dx**2 + top[1]*Alpha/dy**2
                    elif top[0] == 'N' and left[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + Alpha/dy**2
                        source[i,j] = Tc_old[j,i]/dt + left[1]/dx - top[1]/dy
                    elif top[0] == 'D' and left[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[i,j] = Tc_old[j,i]/dt + left[1]/dx + top[1]*Alpha/dy**2
                    elif top[0] == 'N' and left[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[i,j] = Tc_old[j,i]/dt + left[1]*Alpha/dx**2 - top[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = 0.
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = 0.
                    ab[j,i] = Alpha/dy**2
                    
                    
                elif i>imin and i<imax: # boundary nodes:
                    
                    if top[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + top[1]*Alpha/dy**2
                    elif top[0] == 'N':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - top[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                        
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = 0.
                    ab[j,i] = Alpha/dy**2
                    
                    
                elif i==imax: # right corner node:
                    
                    if top[0] == 'D' and right[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + right[1]*Alpha/dx**2 + top[1]*Alpha/dy**2
                    elif top[0] == 'N' and right[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - right[1]/dx - top[1]/dy
                    elif top[0] == 'D' and right[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - right[1]/dx + top[1]*Alpha/dy**2
                    elif top[0] == 'N' and right[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + right[1]*Alpha/dx**2 - top[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = Alpha/dy**2
                     

        elif j==jmax: # bottom row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left corner node:
                    
                    if bottom[0] == 'D' and left[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]*Alpha/dx**2 + bottom[1]*Alpha/dy**2
                    elif bottom[0] == 'N' and left[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]/dx + bottom[1]/dy
                    elif bottom[0] == 'D' and left[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]/dx + bottom[1]*Alpha/dy**2
                    elif bottom[0] == 'N' and left[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]*Alpha/dx**2 + bottom[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = 0.
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = 0.
                    
                elif i>imin and i<imax: # boundary nodes:
                    
                    if bottom[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + bottom[1]*Alpha/dy**2
                    elif bottom[0] == 'N':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + bottom[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = 0.
                    
                elif i==imax: # right corner node:
                    
                    if bottom[0] == 'D' and right[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + right[1]*Alpha/dx**2 + bottom[1]*Alpha/dy**2
                    elif bottom[0] == 'N' and right[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - right[1]/dx + bottom[1]/dy
                    elif bottom[0] == 'D' and right[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - right[1]/dx + bottom[1]*Alpha/dy**2
                    elif bottom[0] == 'N' and right[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + right[1]*Alpha/dx**2 + bottom[1]/dy
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = 0.
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = 0.

        else: # internal rows
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left boundary node:
                    
                    if left[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]*Alpha/dx**2
                    elif left[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + left[1]/dx
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = 0.
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = Alpha/dy**2
                    
                elif i>imin and i<imax: # internal nodes:
                    ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = Alpha/dx**2
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = Alpha/dy**2
                    source[j,i] = Tc_old[j,i]/dt
                    
                elif i==imax: # right boundary node:
                    
                    if right[0] == 'D':
                        ac[j,i] = (1./dt) + 2*Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt + right[1]*Alpha/dx**2
                    elif right[0] == 'N':
                        ac[j,i] = (1./dt) + Alpha/dx**2 + 2*Alpha/dy**2
                        source[j,i] = Tc_old[j,i]/dt - right[1]/dx
                    else:
                        raise ValueError('Appropriate boundary conditions not chosen')
                    
                    al[j,i] = Alpha/dx**2
                    ar[j,i] = 0.
                    at[j,i] = Alpha/dy**2
                    ab[j,i] = Alpha/dy**2
                    
    
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])

def solve_Temperature2D(K,Cp,rho,ny,nx,dx,dy,left,top,right,bottom,file_name,path_name,T_ini,dt,nt,save_t):
    
    t = 0.
    T_old = T_ini.copy()
        
    while t <= nt:
        
        # Boundary Conditions implementation over the grid:
        if left[0] == 'D' and top[0] == 'D' and right[0] == 'D' and bottom[0] == 'D':
            T_old[:,0] = left[1]
            T_old[0,:] = top[1]
            T_old[:,-1] = right[1]
            T_old[-1,:] = bottom[1]
        elif left[0] == 'N' and top[0] == 'D' and right[0] == 'D' and bottom[0] == 'D':
            T_old[0,:] = top[1]
            T_old[:,-1] = right[1]
            T_old[-1,:] = bottom[1]
            T_old[:,0] = T_old[:,1]+left[1]*dx
        else:
            raise ValueError('This boundary condition is not yet implemented')
        
        #print(np.abs(t-save_t),'\n')
        if np.abs(t-save_t)<=0.1 or t==0.:
            print('############################ Time = ',int(t)+1,' sec. ############################','\n')
            data = {}
            data['T'] = T_old.flatten()
            csv.csv_fileWriter(path_name, file_name+str(int(t)+1)+'.csv', ',', data)
            if t>0:
                save_t += 10.
        
        A, S = assemble_A_and_S(K,Cp,rho,T_old,dt,ny,nx,dx,dy,left,top,right,bottom)
        sol = np.linalg.solve(A,S.flatten())
        sol = np.reshape(sol,(ny-2,nx-2))
        T_old[1:-1,1:-1] = sol[:,:]
        
        t = t+dt


