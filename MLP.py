#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:41:24 2020

@author: sn249179
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:37:48 2020
@author: snandan
"""

import numpy as np
import CSV_FileReader_Writer as csv
import os


def Grad_X(Sc, j, i, h): # Sc = value of a scalar S at the cell centers
    
    grad = (1./12.)*(Sc[j+1,i+1] - Sc[j+1,i-1]) + (1./3.)*(Sc[j,i+1] - Sc[j,i-1]) + (1./12.)*(Sc[j-1,i+1] - Sc[j-1,i-1])
        
    return grad/h # returns 8-point x-gradient for any cell located at (j,i) index 

def Grad_Y(Sc, j, i, h): # Sc = value of a scalar S at the cell centers
    
    grad = (1./12.)*(Sc[j-1,i+1] - Sc[j+1,i+1]) + (1./3.)*(Sc[j-1,i] - Sc[j+1,i]) + (1./12.)*(Sc[j-1,i-1] - Sc[j+1,i-1])
    
    return grad/h # returns 8-point y-gradient for any cell located at (j,i) index

def Sc_max(Sc, cr_index): # examines all the cells surrounding any corner located at (j,i) index, and returns it's max value
    j,i = cr_index
    return np.max((Sc[j+1,i+1], Sc[j+1,i+2], Sc[j+2,i+1], Sc[j+2,i+2]))

def Sc_min(Sc, cr_index):# examines all the cells surrounding any corner located at (j,i) index, and returns it's min value
    j,i = cr_index
    return np.min((Sc[j+1,i+1], Sc[j+1,i+2], Sc[j+2,i+1], Sc[j+2,i+2]))

def Sc_extremum(S_NL, Sc, c_index, cr_index):
    extremum = 0.
    if S_NL > Sc[c_index]:
        extremum = Sc_max(Sc, cr_index)
    elif S_NL < Sc[c_index]:
        extremum = Sc_min(Sc, cr_index)
    else:
        extremum = Sc[c_index]
    
    return extremum
        
def MLP_Limiter(Sc, dx, dy, beta, epsilon):
    
    limiter_values_centers = np.zeros(np.shape(Sc))
    
    # loops over cells:
    for j in np.flip(range(2,np.shape(Sc)[0]-2)): # loop over rows in 2D cartesian structured grid
        for i in range(2,np.shape(Sc)[0]-2): # loop over columns in 2D cartesian structured grid
            
            left_top_cr = (j-2,i-2)
            left_bottom_cr = (j-1,i-2)
            right_top_cr = (j-2,i-1)
            right_bottom_cr = (j-1,i-1)
            
            
            S_NL_left_top_cr = Sc[j,i] - Grad_X(Sc,j,i,dx) * (dx/2.) + Grad_Y(Sc,j,i,dy) * (dy/2.)
            
            S_NL_left_bottom_cr = Sc[j,i] - Grad_X(Sc,j,i,dx) * (dx/2.) - Grad_Y(Sc,j,i,dy) * (dy/2.)
            
            S_NL_right_top_cr = Sc[j,i] + Grad_X(Sc,j,i,dx) * (dx/2.) + Grad_Y(Sc,j,i,dy) * (dy/2.)
            
            S_NL_right_bottom_cr = Sc[j,i] + Grad_X(Sc,j,i,dx) * (dx/2.) - Grad_Y(Sc,j,i,dy) * (dy/2.)
            
            
            extremum_Scr_left_top_cr = Sc_extremum(S_NL_left_top_cr, Sc, (j,i), left_top_cr)
            
            extremum_Scr_left_bottom_cr = Sc_extremum(S_NL_left_bottom_cr, Sc, (j,i), left_bottom_cr)
            
            extremum_Scr_right_top_cr = Sc_extremum(S_NL_right_top_cr, Sc, (j,i), right_top_cr)
            
            extremum_Scr_right_bottom_cr = Sc_extremum(S_NL_right_bottom_cr, Sc, (j,i), right_bottom_cr)
            
            
            diff_NL_Sc_left_top_cr = S_NL_left_top_cr - Sc[j,i]
            
            diff_NL_Sc_left_bottom_cr = S_NL_left_bottom_cr - Sc[j,i]
            
            diff_NL_Sc_right_top_cr = S_NL_right_top_cr - Sc[j,i]
            
            diff_NL_Sc_right_bottom_cr = S_NL_right_bottom_cr - Sc[j,i]
            
            
            diff_exm_Sc_left_top_cr = extremum_Scr_left_top_cr - Sc[j,i]
            
            diff_exm_Sc_left_bottom_cr = extremum_Scr_left_bottom_cr - Sc[j,i]
            
            diff_exm_Sc_right_top_cr = extremum_Scr_right_top_cr - Sc[j,i]
            
            diff_exm_Sc_right_bottom_cr = extremum_Scr_right_bottom_cr - Sc[j,i]
            
            # ensuring boundedness
            if S_NL_left_top_cr == Sc[j,i]:
                diff_NL_Sc_left_top_cr = epsilon
                
            if S_NL_left_bottom_cr == Sc[j,i]:
                diff_NL_Sc_left_bottom_cr = epsilon
                
            if S_NL_right_top_cr == Sc[j,i]:
                diff_NL_Sc_right_top_cr = epsilon
            
            if S_NL_right_bottom_cr == Sc[j,i]:
                diff_NL_Sc_right_bottom_cr = epsilon
            
            
            limiter_values_left_top_cr = diff_exm_Sc_left_top_cr / diff_NL_Sc_left_top_cr
            limiter_values_left_top_cr = np.min((beta, limiter_values_left_top_cr))
            
            limiter_values_left_bottom_cr = diff_exm_Sc_left_bottom_cr / diff_NL_Sc_left_bottom_cr
            limiter_values_left_bottom_cr = np.min((beta, limiter_values_left_bottom_cr))
            
            limiter_values_right_top_cr = diff_exm_Sc_right_top_cr / diff_NL_Sc_right_top_cr
            limiter_values_right_top_cr = np.min((beta, limiter_values_right_top_cr))
            
            limiter_values_right_bottom_cr = diff_exm_Sc_right_bottom_cr / diff_NL_Sc_right_bottom_cr
            limiter_values_right_bottom_cr = np.min((beta, limiter_values_right_bottom_cr))
            
            
            limiter_values_centers[j,i] = np.min((limiter_values_left_top_cr, limiter_values_left_bottom_cr, \
                                                 limiter_values_right_top_cr, limiter_values_right_bottom_cr))
            
            # if limiter_values_centers[j,i]< 0.:
            #     print('X limiter values less than 0', limiter_values_centers[j,i])  
            #     print(diff_NL_Sc_left_top_cr,'\n')
            #     print(diff_NL_Sc_left_bottom_cr,'\n')
            #     print(diff_NL_Sc_right_top_cr,'\n')
            #     print(diff_NL_Sc_right_bottom_cr,'\n')
            #     print(limiter_values_left_top_cr,'\n')
            #     print(limiter_values_left_bottom_cr,'\n')
            #     print(limiter_values_right_top_cr,'\n')
            #     print(limiter_values_right_bottom_cr,'\n')
            #     print((j,i))
    
    return limiter_values_centers
    
def Face_Flux(S_LB, S_RT): #Gudonov type flux determination
    
    sFace = 0.
    
    if S_LB > 0. and S_RT > 0.: # classical Up-Wind
        sFace = S_LB
    elif S_LB < 0. and S_RT < 0.: # classical Up-Wind
        sFace = S_RT
    elif S_LB < 0. and S_RT > 0.: # fanning condition
        sFace = 0.
    elif S_LB > 0. and S_RT < 0.: # shock propagation condition
        c_shock = (S_LB+S_RT)/2. 
        if c_shock > 0.:
            sFace = S_LB
        elif c_shock < 0.:
            sFace = S_RT
    elif S_LB == 0. and S_RT > 0.:
        sFace = 0.
    elif S_LB > 0. and S_RT == 0.:
        sFace = S_LB
    
    return sFace
    
def solve_linearAdvection2D(c,beta,dx,dy,left,top,right,bottom,file_name,path_name,Sc,dt,t,save_t,epsilon):
        
    nt = int(t/dt)
    save_n = int(save_t/dt)
    
    file_num = 0
    
    if left[0] != 'D' and top[0] != 'D' and right[0] != 'D' and bottom[0] != 'D':
        raise ValueError('Only Dirichlet boundary conditions are applicable for current case of Linear 2D Advection')
        exit()
        
    for n in range(0, nt+1):
        
        Sc_old = Sc.copy()
        
        # Boundary Conditions implementation over the grid:
        Sc_old[:,0:2] = left[1]
        Sc_old[0:2,:] = top[1]
        Sc_old[:,-1] = right[1]
        Sc_old[:,-2] = right[1]
        Sc_old[-1,:] = bottom[1]
        Sc_old[-2,:] = bottom[1]
        
        #print(np.abs(t-save_t),'\n')
        if np.abs(n-save_n) == 0 or n == 0:
            print('############################ Time = ',n*dt,' sec. ############################','\n')
            data = {}
            data['Sc'] = Sc_old.flatten()
            csv.csv_fileWriter(path_name, file_name+str(file_num)+'.csv', ',', data)
            if n>0:
                save_n += int(save_t/dt)
            file_num += 1
        
        mlp_limiter = MLP_Limiter(Sc_old, dx, dy, beta, epsilon)
        
        # if np.any(mlp_limiter_x< 0.):
        #     print('X limiter values less than 0', mlp_limiter_x[mlp_limiter_x< 0.])
        #     break
            
    
        # loops over cells:
        for j in np.flip(range(2,np.shape(Sc)[0]-2)): # loop over rows in 2D cartesian structured grid
            for i in range(2,np.shape(Sc)[0]-2): # loop over columns in 2D cartesian structured grid
                
                # MUSCL type linear reconstruction in each cell using MLP type limiter
                Sw_L = Sc_old[j,i-1] + mlp_limiter[j,i-1]*Grad_X(Sc_old, j, i-1, dx)*(dx/2.)
                
                Sw_R = Sc_old[j,i] - mlp_limiter[j,i]*Grad_X(Sc_old, j, i, dx)*(dx/2.)
                
                Se_L = Sc_old[j,i] + mlp_limiter[j,i]*Grad_X(Sc_old, j, i, dx)*(dx/2.)
                
                Se_R = Sc_old[j,i+1] - mlp_limiter[j,i+1]*Grad_X(Sc_old, j, i+1, dx)*(dx/2.)
                
                Ss_B = Sc_old[j+1,i] + mlp_limiter[j+1,i]*Grad_Y(Sc_old, j+1, i, dy)*(dy/2.)
                
                Ss_T = Sc_old[j,i] - mlp_limiter[j,i]*Grad_Y(Sc_old, j, i, dy)*(dy/2.)
                
                Sn_B = Sc_old[j,i] + mlp_limiter[j,i]*Grad_Y(Sc_old, j, i, dy)*(dy/2.)
                
                Sn_T = Sc_old[j-1,i] - mlp_limiter[j-1,i]*Grad_Y(Sc_old, j-1, i, dy)*(dy/2.)
                
                
                # Explicit R-K2 time stepping with Gudonow type flux determination
                
                K1 = (c/dx)*(Face_Flux(Sw_L, Sw_R) - Face_Flux(Se_L, Se_R)) + (c/dy)*(Face_Flux(Ss_B, Ss_T) - Face_Flux(Sn_B, Sn_T))
                
                K2 = (c/dx)*((Face_Flux(Sw_L, Sw_R)+K1*dt) - (Face_Flux(Se_L, Se_R)+K1*dt)) + (c/dy)*((Face_Flux(Ss_B, Ss_T)+K1*dt) - (Face_Flux(Sn_B, Sn_T)+K1*dt))
                
                Sc[j,i] = Sc_old[j,i] + (0.5*K1 + 0.5*K2)*dt
                
                # Explicit Euler time stepping with Gudonow type flux determination
                #Sc[j,i] = Sc_old[j,i] + (c*dt/dx)*(Face_Flux(Sw_L, Sw_R) - Face_Flux(Se_L, Se_R)) + (c*dt/dy)*(Face_Flux(Ss_B, Ss_T) - \
                #                                                                                          Face_Flux(Sn_B, Sn_T))
    



if __name__ == "__main__":
    
    nx = 512 + 4 # total number of cells with 2 set of ghost cells on all the boundaries
    ny = 512 + 4
    c = 1.
    dx = 2/(nx-4)
    dy = 2/(ny-4)
    dt = 0.00117
    beta = 2.
    t = 0.5 # with these dx,dt and c, the CFL number is = 0.3
    
    Sc_ini = np.zeros((ny,nx))
    
    # Boundary conditions:
    left = ('D', 0.0)
    top = ('D', 0.0)
    right = ('D', 0.0)
    bottom = ('D', 0.0)
    
    ## set hat function I.C.
    Sc_ini[int(1.3 / dy):int(1.8 / dy + 1.8), int(.5 / dx):int(1. / dx + 1.)] = 1.
    
    file_name = '/squareAdvection_explicitEuler_MLP'
    path_name = os.getcwd()
    
    save_t = 0.01
    
    epsilon = 1.e-11
    
    solve_linearAdvection2D(c,beta,dx,dy,left,top,right,bottom,file_name,path_name,Sc_ini,dt,t,save_t,epsilon)
    
    