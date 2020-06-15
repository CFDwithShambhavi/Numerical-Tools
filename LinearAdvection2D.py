#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:02:24 2020

@author: snandan
"""

import numpy as np
import CSV_FileReader_Writer as csv
import MatrixTools as mt
#import os


def assemble_A_and_S(c,uc_old,dt,ny,nx,dx,dy,left,top,right,bottom,num_scheme):
    
    uc = uc_old.copy()
    
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
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[i,j] = uc_old[j,i] + left[1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[i,j] = uc_old[j,i] + left[1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - top[1]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0. 
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
                elif i>imin and i<imax: # boundary nodes:
                    
                    if top[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - top[1]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
                elif i==imax: # right corner node:
                    
                    if top[0] == 'D' and right[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - right[1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - top[1]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.

        elif j==jmax: # bottom row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left corner node:
                    
                    if bottom[0] == 'D' and left[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + left[1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + bottom[1]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + left[1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + bottom[1]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
                elif i>imin and i<imax: # boundary nodes:
                    
                    if bottom[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + bottom[1]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + bottom[1]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
                elif i==imax: # right corner node:
                    
                    if bottom[0] == 'D' and right[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + bottom[1]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - right[1]*(c*dt/dx/2.) + bottom[1]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.

        else: # internal rows
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left boundary node:
                    
                    if left[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + left[1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + left[1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
                elif i>imin and i<imax: # internal nodes:
                    ac[j,i] = 1.
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    if num_scheme == 'FTBS' or num_scheme == 'UW':
                        source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                    elif num_scheme == 'CDS':
                        source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - uc[j,i+1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                    else:
                        raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    
                elif i==imax: # right boundary node:
                    
                    if right[0] == 'D':
                        ac[j,i] = 1.
                        if num_scheme == 'FTBS' or num_scheme == 'UW':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx) - uc[j,i]*(c*dt/dx) + uc[j+1,i]*(c*dt/dy) - uc[j,i]*(c*dt/dy)
                        elif num_scheme == 'CDS':
                            source[j,i] = uc_old[j,i] + uc[j,i-1]*(c*dt/dx/2.) - right[1]*(c*dt/dx/2.) + uc[j+1,i]*(c*dt/dy/2.) - uc[j-1,i]*(c*dt/dy/2.)
                        else:
                            raise ValueError('The input numerical scheme is not applicable for case of Linear Advection')
                    else:
                        raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                    
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    
    
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])


def solve_linearAdvection2D(c,ny,nx,dx,dy,left,top,right,bottom,file_name,path_name,u_ini,dt,nt,save_t,num_scheme):
    
    t = 0.
    u_old = u_ini.copy()
        
    while t <= nt:
        
        # Boundary Conditions implementation over the grid:
        if left[0] == 'D' and top[0] == 'D' and right[0] == 'D' and bottom[0] == 'D':
            u_old[:,0] = left[1]
            u_old[0,:] = top[1]
            u_old[:,-1] = right[1]
            u_old[-1,:] = bottom[1]
        else:
            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
        
        #print(np.abs(t-save_t),'\n')
        if np.abs(t-save_t)<=0.1 or t==0.:
            print('############################ Time = ',int(t)+1,' sec. ############################','\n')
            data = {}
            data['u'] = u_old.flatten()
            csv.csv_fileWriter(path_name, file_name+str(int(t)+1)+'.csv', ',', data)
            if t>0:
                save_t += 10.
        
        A, S = assemble_A_and_S(c,u_old,dt,ny,nx,dx,dy,left,top,right,bottom,num_scheme)
        sol = np.linalg.solve(A,S.flatten())
        sol = np.reshape(sol,(ny-2,nx-2))
        u_old[1:-1,1:-1] = sol[:,:]
        
        t = t+dt

