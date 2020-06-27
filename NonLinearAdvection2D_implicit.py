#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:11:52 2020

@author: snandan
"""
import numpy as np
import CSV_FileReader_Writer as csv
import MatrixTools as mt
np.seterr(divide='ignore', invalid='ignore')
#import os

def assemble_A_and_S_for_u(uc_old,u,v,rho,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme):
    
    # Scalar variable datastructure:
    ac = np.zeros((ny,nx))
    al = np.zeros((ny,nx))
    ar = np.zeros((ny,nx))
    at = np.zeros((ny,nx))
    ab = np.zeros((ny,nx))
    source = np.zeros((ny,nx))
    
    if time_scheme == 'implicit' and space_scheme == 'UW':
        
        jmin = 1
        jmax = ny-2
        for j in np.arange(jmin, jmax+1):
            if j==jmin: # top row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==1: # left corner node:
                        
                        if top[0] == 'D' and left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = uc_old[j,i]*rho + left[1]*np.fmax(fw,0.)*dt/dx + top[1]*np.fmax(-fn,0.)*dt/dy
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if top[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = uc_old[j,i]*rho + top[1]*np.fmax(-fn,0.)*dt/dy
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                            
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i==imax: # right corner node:
                        
                        if top[0] == 'D' and right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = uc_old[j,i]*rho + right[1]*np.fmax(-fe,0.)*dt/dx + top[1]*np.fmax(-fn,0.)*dt/dy
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        
                        if bottom[0] == 'D' and left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = uc_old[j,i]*rho + left[1]*np.fmax(fw,0.)*dt/dx + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if bottom[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = uc_old[j,i]*rho + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
                        
                    elif i==imax: # right corner node:
                        
                        if bottom[0] == 'D' and right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = uc_old[j,i]*rho + right[1]*np.fmax(-fe,0.)*dt/dx + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        
                        if left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)  
                            
                            source[j,i] = uc_old[j,i]*rho + left[1]*np.fmax(fw,0.)*dt/dx
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i>imin and i<imax: # internal nodes:
                        cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                        cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                        
                        if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                            fe = rho*u[j,i]
                            fw = rho*u[j,i-1]
                            fn = rho*v[j,i]
                            fs = rho*v[j+1,i]
                        else:                    # wave speed in x-y-direction from right to left and top to bottom
                            fe = rho*u[j,i+1]
                            fw = rho*u[j,i]
                            fn = rho*v[j-1,i]
                            fs = rho*v[j,i]
                        
                        ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                        source[j,i] = uc_old[j,i]*rho
                        
                    elif i==imax: # right boundary node:
                        
                        if right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.: # wave speed in x-y-direction from left to right and bottom to top
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:                    # wave speed in x-y-direction from right to left and top to bottom
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = uc_old[j,i]*rho + right[1]*np.fmax(-fe,0.)*dt/dx
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
    else:
        raise ValueError('For nonlinear PDE only "implicit" time scheme and "UW" space scheme is valid')
        
        
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])

#........................................................................................................................................................

def assemble_A_and_S_for_v(vc_old,u,v,rho,dt,ny,nx,dx,dy,left,top,right,bottom,space_scheme,time_scheme):
    
    # Scalar variable datastructure:
    ac = np.zeros((ny,nx))
    al = np.zeros((ny,nx))
    ar = np.zeros((ny,nx))
    at = np.zeros((ny,nx))
    ab = np.zeros((ny,nx))
    source = np.zeros((ny,nx))
    
    if time_scheme == 'implicit' and space_scheme == 'UW':
        
        jmin = 1
        jmax = ny-2
        for j in np.arange(jmin, jmax+1):
            if j==jmin: # top row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==1: # left corner node:
                        
                        if top[0] == 'D' and left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = vc_old[j,i]*rho + left[1]*(np.fmax(fw,0.)*dt/dx) + top[1]*(np.fmax(-fn,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if top[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = vc_old[j,i]*rho + top[1]*(np.fmax(-fn,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                            
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i==imax: # right corner node:
                        
                        if top[0] == 'D' and right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = vc_old[j,i]*rho + right[1]*(np.fmax(-fe,0.)*dt/dx) + top[1]*(np.fmax(-fn,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = 0.
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        
                        if bottom[0] == 'D' and left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)
                            
                            source[j,i] = vc_old[j,i]*rho + left[1]*(np.fmax(fw,0.)*dt/dx) + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
                        
                    elif i>imin and i<imax: # boundary nodes:
                        
                        if bottom[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = vc_old[j,i]*rho + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
                        
                    elif i==imax: # right corner node:
                        
                        if bottom[0] == 'D' and right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)  
                            
                            source[j,i] = vc_old[j,i]*rho + right[1]*(np.fmax(-fe,0.)*dt/dx) + bottom[1]*(np.fmax(fs,0.)*dt/dy)
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = 0.
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        
                        if left[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                            
                            source[j,i] = vc_old[j,i]*rho + left[1]*(np.fmax(fw,0.)*dt/dx) 
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = 0.
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                    elif i>imin and i<imax: # internal nodes:
                        cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                        cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                        
                        if cx >= 0. and cy >= 0.:
                            fe = rho*u[j,i]
                            fw = rho*u[j,i-1]
                            fn = rho*v[j,i]
                            fs = rho*v[j+1,i]
                        else:
                            fe = rho*u[j,i+1]
                            fw = rho*u[j,i]
                            fn = rho*v[j-1,i]
                            fs = rho*v[j,i]
                            
                        ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.) 
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = np.fmax(-fe,0.)*dt/dx
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
                        source[j,i] = vc_old[j,i]*rho
                        
                    elif i==imax: # right boundary node:
                        
                        if right[0] == 'D':
                            cx = (u[j,i] + u[j,i+1])/2. # wave speed in x-direction
                            cy = (v[j,i] + v[j-1,i])/2. # wave speed in y-direction
                            
                            if cx >= 0. and cy >= 0.:
                                fe = rho*u[j,i]
                                fw = rho*u[j,i-1]
                                fn = rho*v[j,i]
                                fs = rho*v[j+1,i]
                            else:
                                fe = rho*u[j,i+1]
                                fw = rho*u[j,i]
                                fn = rho*v[j-1,i]
                                fs = rho*v[j,i]
                            
                            ac[j,i] = rho + (dt/dx)*np.fmax(fe,0.) + (dt/dx)*np.fmax(-fw,0.) + (dt/dy)*np.fmax(fn,0.) + (dt/dy)*np.fmax(-fs,0.)  
                            
                            source[j,i] = vc_old[j,i]*rho + right[1]*(np.fmax(-fe,0.)*dt/dx)  
                        else:
                            raise ValueError('Only Dirichlet boundary conditions are applicable for case of Non-Linear Advection')
                        
                        al[j,i] = np.fmax(fw,0.)*dt/dx
                        ar[j,i] = 0.
                        at[j,i] = np.fmax(-fn,0.)*dt/dy
                        ab[j,i] = np.fmax(fs,0.)*dt/dy
                        
    else:
        raise ValueError('For nonlinear PDE only "implicit" time scheme and "UW" space scheme is valid')
        
        
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    
    return (A, source[1:-1,1:-1])

#........................................................................................................................................................



def solve_nonlinearAdvection2D(ny,nx,dx,dy,u_bc,v_bc,file_name,path_name,u_ini,v_ini,rho,dt,t,save_t,space_scheme,time_scheme):
    
    u_left = u_bc['u_left']
    u_top = u_bc['u_top']
    u_right = u_bc['u_right']
    u_bottom = u_bc['u_bottom']
    
    v_left = v_bc['v_left']
    v_top = v_bc['v_top']
    v_right = v_bc['v_right']
    v_bottom = v_bc['v_bottom']
    
    u_old = u_ini.copy()
    v_old = v_ini.copy()
    
    u = u_ini.copy()
    v = v_ini.copy()
    
    e = 1.0e-3 # tolerance for convergence within loop tackling the non-linearity
    nt = int(t/dt)
    save_n = int(save_t/dt)
    
    file_num = 0
    
    for n in range(0, nt+1):
        
        # Boundary Conditions implementation over the grid:
        if u_left[0] == 'D' and u_top[0] == 'D' and u_right[0] == 'D' and u_bottom[0] == 'D':
            u_old[:,0] = u_left[1]
            u_old[0,:] = u_top[1]
            u_old[:,-1] = u_right[1]
            u_old[-1,:] = u_bottom[1]
            
            v_old[:,0] = v_left[1]
            v_old[0,:] = v_top[1]
            v_old[:,-1] = v_right[1]
            v_old[-1,:] = v_bottom[1]
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
        
        iter_conv = 0
        # Non-linearity loop.........................................................................................
        while True:
        
            Au, Su = assemble_A_and_S_for_u(u_old,u,v,rho,dt,ny,nx,dx,dy,u_left,u_top,u_right,u_bottom,space_scheme,time_scheme)
            
            solu = np.linalg.solve(Au,Su.flatten())
            solu = np.reshape(solu,(ny-2,nx-2))
            
            Av, Sv = assemble_A_and_S_for_v(v_old,u,v,rho,dt,ny,nx,dx,dy,v_left,v_top,v_right,v_bottom,space_scheme,time_scheme)
            
            solv = np.linalg.solve(Av,Sv.flatten())
            solv = np.reshape(solv,(ny-2,nx-2))
        
            #print(np.sum(np.abs(u[1:-1,1:-1]-solu))/((ny-2)*(nx-2)),'\n')
            if np.all(np.abs(u[1:-1,1:-1]-solu) <= e) and np.all(np.abs(v[1:-1,1:-1]-solv) <= e):
                print('******** Convergence in ',iter_conv,' iterations ********','\n')
                u[1:-1,1:-1] = solu[:,:]
                v[1:-1,1:-1] = solv[:,:]
                break
            elif iter_conv >= 2000:
                print('******** Convergence fail ********','\n')
                break
            else:
                u[1:-1,1:-1] = solu[:,:]
                v[1:-1,1:-1] = solv[:,:]
            
            iter_conv += 1
            print(iter_conv,'\n')
        #............................................................................................................
        
        u_old[1:-1,1:-1] = u[1:-1,1:-1]
        v_old[1:-1,1:-1] = v[1:-1,1:-1]
        