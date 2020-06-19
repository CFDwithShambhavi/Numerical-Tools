#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:18:21 2020

@author: snandan
"""

import numpy as np
import CSV_FileReader_Writer as csv
import MatrixTools as mt
#import os


def assemble_A_and_S_for_u(uc_old,vc_old,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme):
    
    uc = uc_old.copy()
    vc = vc_old.copy()
    
    # Scalar variable datastructure:
    ac = np.zeros((ny,nx))
    al = np.zeros((ny,nx))
    ar = np.zeros((ny,nx))
    at = np.zeros((ny,nx))
    ab = np.zeros((ny,nx))
    source = np.zeros((ny,nx))
    
    if time_scheme == 'explicit' and space_scheme == 'UW':
        
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
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if top[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i==imax: # right corner node:
                        
                        if top[0] == 'D' and right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        
                        if bottom[0] == 'D' and left[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if bottom[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i==imax: # right corner node:
                        
                        if bottom[0] == 'D' and right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        
                        if left[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # internal nodes:
                        ac[j,i] = 1.
                        source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        
                    elif i==imax: # right boundary node:
                        
                        if right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = uc_old[j,i] - (uc[j,i]*dt/dx)*uc[j,i] + (uc[j,i]*dt/dx)*uc[j,i-1] - (vc[j,i]*dt/dy)*uc[j,i] + (vc[j,i]*dt/dy)*uc[j-1,i]
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
    else:
        raise ValueError('For nonlinear PDE only "explicit" time scheme and "UW" space scheme is valid')
        
        
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])


#........................................................................................................................................................

def assemble_A_and_S_for_v(uc_old,vc_old,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme):
    
    uc = uc_old.copy()
    vc = vc_old.copy()
    
    # Scalar variable datastructure:
    ac = np.zeros((ny,nx))
    al = np.zeros((ny,nx))
    ar = np.zeros((ny,nx))
    at = np.zeros((ny,nx))
    ab = np.zeros((ny,nx))
    source = np.zeros((ny,nx))
    
    if time_scheme == 'explicit' and space_scheme == 'UW':
        
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
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if top[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i==imax: # right corner node:
                        
                        if top[0] == 'D' and right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        
                        if bottom[0] == 'D' and left[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if bottom[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i==imax: # right corner node:
                        
                        if bottom[0] == 'D' and right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        
                        if left[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
                    elif i>imin and i<imax: # internal nodes:
                        ac[j,i] = 1.
                        source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        
                    elif i==imax: # right boundary node:
                        
                        if right[0] == 'D':
                            ac[j,i] = 1.
                            source[j,i] = vc_old[j,i] - uc[j,i]*(vc[j,i]*dt/dx) + uc[j,i]*(vc[j,i-1]*dt/dx) - vc[j,i]*(vc[j,i]*dt/dy) + vc[j,i]*(vc[j-1,i]*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Linear Advection')
                        
    else:
        raise ValueError('For nonlinear PDE only "explicit" time scheme and "UW" space scheme is valid')
        
        
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])


#........................................................................................................................................................



def solve_nonlinearAdvection2D(ny,nx,dx,dy,left,top,right,bottom,file_name,path_name,u_ini,v_ini,dt,t,save_t,space_scheme,time_scheme):
    
    u_old = u_ini.copy()
    v_old = v_ini.copy()
    
    nt = int(t/dt)
    save_n = int(save_t/dt)
    
    file_num = 0
        
    for n in range(0, nt+1):
        
        # Boundary Conditions implementation over the grid:
        if left[0] == 'D' and top[0] == 'D' and right[0] == 'D' and bottom[0] == 'D':
            u_old[:,0] = left[1]
            u_old[0,:] = top[1]
            u_old[:,-1] = right[1]
            u_old[-1,:] = bottom[1]
            
            v_old[:,0] = left[1]
            v_old[0,:] = top[1]
            v_old[:,-1] = right[1]
            v_old[-1,:] = bottom[1]
        else:
            raise ValueError('Only Dirichlet boundary conditions are applicable for case of 2D Advection')
            
        if np.abs(n-save_n) == 0 or n == 0:
            print('############################ Time = ',n*dt,' sec. ############################','\n')
            data = {}
            data['u'] = u_old.flatten()
            data['v'] = v_old.flatten()
            csv.csv_fileWriter(path_name, file_name+str(file_num)+'.csv', ',', data)
            if n>0:
                save_n += int(save_t/dt)
            file_num += 1
        
        Au, Su = assemble_A_and_S_for_u(u_old,v_old,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme)
        
        solu = np.linalg.solve(Au,Su.flatten())
        solu = np.reshape(solu,(ny-2,nx-2))
        
        Av, Sv = assemble_A_and_S_for_v(u_old,v_old,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme)
        
        solv = np.linalg.solve(Av,Sv.flatten())
        solv = np.reshape(solv,(ny-2,nx-2))
        
        u_old[1:-1,1:-1] = solu[:,:]
        v_old[1:-1,1:-1] = solv[:,:]
        
        
    

