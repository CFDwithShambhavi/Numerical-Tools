# Author: Shambhavi Nandan

# This code implements 2D diffusion model for pandemic spread. The original paper can be found her: https://api.semanticscholar.org/CorpusID:9108055.oa

import matplotlib.pyplot as plt
import numpy as np

# For main code skip "Lambda functions" and "assemble_a_toA()" function:

i_j_to_k = lambda index, columns: index[0] * columns + index[1]

R = lambda index: (index[0], index[1] + 1)

L = lambda index: (index[0], index[1] - 1)

T = lambda index: (index[0] - 1, index[1])

B = lambda index: (index[0] + 1, index[1])

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

# Maine code implementing the model starts from here:

def solve_Succeptibility(Ds,Ic_old,Sc_old,Ac_old,dt,ny,nx):
    
    # Succeptibility datastructure:
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
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = 0.
                    ar[j,i] = Ds/dx**2
                    at[j,i] = 0.
                    ab[j,i] = Ds/dx**2
                    source[i,j] = Sc_old[j,i]/dt
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = Ds/dx**2
                    at[j,i] = 0.
                    ab[j,i] = Ds/dx**2
                    source[j,i] = Sc_old[j,i]/dt
                elif i==imax: # right corner node:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = Ds/dx**2
                    source[j,i] = Sc_old[j,i]/dt

        elif j==jmax: # bottom row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left corner node:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = 0.
                    ar[j,i] = Ds/dx**2
                    at[j,i] = Ds/dx**2
                    ab[j,i] = 0.
                    source[j,i] = Sc_old[j,i]/dt
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = Ds/dx**2
                    at[j,i] = Ds/dx**2
                    ab[j,i] = 0.
                    source[j,i] = Sc_old[j,i]/dt
                elif i==imax: # right corner node:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = 0.
                    at[j,i] = Ds/dx**2
                    ab[j,i] = 0.
                    source[j,i] = Sc_old[j,i]/dt

        else: # internal rows
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left boundary node:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = 0.
                    ar[j,i] = Ds/dx**2
                    at[j,i] = Ds/dx**2
                    ab[j,i] = Ds/dx**2
                    source[j,i] = Sc_old[j,i]/dt
                elif i>imin and i<imax: # internal nodes:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = Ds/dx**2
                    at[j,i] = Ds/dx**2
                    ab[j,i] = Ds/dx**2
                    source[j,i] = Sc_old[j,i]/dt
                elif i==imax: # right boundary node:
                    ac[j,i] = (1./dt) + Ic_old[j,i] + Ac_old[j,i] + 4*Ds/dx**2
                    al[j,i] = Ds/dx**2
                    ar[j,i] = 0.
                    at[j,i] = Ds/dx**2
                    ab[j,i] = Ds/dx**2
                    source[j,i] = Sc_old[j,i]/dt
    
    A = assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
    
    sol = np.reshape(sol,(ny-2,nx-2))
    
    return sol

def solve_Infection(Di,Ic_old,Sc_old,Ac_old,dt,ny,nx):
    
    n_it = 0 # iterator

    e = 1.0e-3 # tolerance for convergence
    
    Ic_guess = np.zeros((ny,nx))
    Ic_guess[:,:] = Ic_old[:,:]
    
    ss = 1.
    
    while True:
        n_it += 1
        # Infection datastructure:
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
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Ds/dx**2
                        at[j,i] = 0.
                        ab[j,i] = Ds/dx**2
                        source[i,j] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # boundary nodes:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = Ds/dx**2
                        at[j,i] = 0.
                        ab[j,i] = Ds/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i==imax: # right corner node:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = 0.
                        at[j,i] = 0.
                        ab[j,i] = Ds/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Ds/dx**2
                        at[j,i] = Ds/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # boundary nodes:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = Ds/dx**2
                        at[j,i] = Ds/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i==imax: # right corner node:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = 0.
                        at[j,i] = Ds/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Ds/dx**2
                        at[j,i] = Ds/dx**2
                        ab[j,i] = Ds/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # internal nodes:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = Ds/dx**2
                        at[j,i] = Ds/dx**2
                        ab[j,i] = Ds/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt 
                    elif i==imax: # right boundary node:
                        ac[j,i] = (1./dt) + ss + 4*Ds/dx**2
                        al[j,i] = Ds/dx**2
                        ar[j,i] = 0.
                        at[j,i] = Ds/dx**2
                        ab[j,i] = Ds/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
        
        A = assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
        sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
        
        sol = np.reshape(sol,(ny-2,nx-2))
        
        print(np.sum(np.abs(Ic_guess[1:-1,1:-1]-sol))/((ny-2)*(nx-2)),'\n')
        
        if np.all(np.abs(Ic_guess[1:-1,1:-1]-sol) <= e) or n_it == 100:
            print(np.sum(np.abs(Ic_guess[1:-1,1:-1]-sol))/((ny-2)*(nx-2)),'\n')
            print('####### Convergence for Infection ########','\n')
            break
        else:
            Ic_guess[1:-1,1:-1] = sol[:,:]
    
    return sol

def solve_Recovery(Ic_old,Rc_old,dt,ny,nx):
    
    # Infection datastructure:
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
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i==imax: # right corner node:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt

        elif j==jmax: # bottom row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left corner node:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i==imax: # right corner node:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt

        else: # internal rows
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left boundary node:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i>imin and i<imax: # internal nodes:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
                elif i==imax: # right boundary node:
                    ac[j,i] = 1./dt
                    al[j,i] = 0.
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 0.
                    source[i,j] = Ic_old[j,i] + Rc_old[j,i]/dt
    
    A = assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
    
    sol = np.reshape(sol,(ny-2,nx-2))
    
    return sol

def solve_Agent(Ic_old,Sc_old,Ac_old,eta,gamma,rho,dt,ny,nx):
    
    # Succeptibility datastructure:
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
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 0.
                    ar[j,i] = 1./dx**2
                    at[j,i] = 0.
                    ab[j,i] = 1./dx**2
                    source[i,j] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 1./dx**2
                    at[j,i] = 0.
                    ab[j,i] = 1./dx**2
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i==imax: # right corner node:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 0.
                    at[j,i] = 0.
                    ab[j,i] = 1./dx**2
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]

        elif j==jmax: # bottom row
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left corner node:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 0.
                    ar[j,i] = 1./dx**2
                    at[j,i] = 1./dx**2
                    ab[j,i] = 0.
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i>imin and i<imax: # boundary nodes:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 1./dx**2
                    at[j,i] = 1./dx**2
                    ab[j,i] = 0.
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i==imax: # right corner node:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 0.
                    at[j,i] = 1./dx**2
                    ab[j,i] = 0.
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]

        else: # internal rows
            imin = 1
            imax = nx-2
            for i in np.arange(imin, imax+1):
                if i==imin: # left boundary node:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 0.
                    ar[j,i] = 1./dx**2
                    at[j,i] = 1./dx**2
                    ab[j,i] = 1./dx**2
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i>imin and i<imax: # internal nodes:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 1./dx**2
                    at[j,i] = 1./dx**2
                    ab[j,i] = 1./dx**2
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
                elif i==imax: # right boundary node:
                    ac[j,i] = (1./dt) + eta*Sc_old[j,i] + gamma + 4/dx**2
                    al[j,i] = 1./dx**2
                    ar[j,i] = 0.
                    at[j,i] = 1./dx**2
                    ab[j,i] = 1./dx**2
                    source[j,i] = Ac_old[j,i]/dt + rho*Ic_old[j,i]
    
    A = assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
    
    sol = np.reshape(sol,(ny-2,nx-2))
    
    return sol


# Grid parameters:
nx = 40
ny = nx
lx = 100
ly = lx
dx = lx / nx
dy = ly / ny


# Succeptibility initialization:
S = np.ones((ny, nx))
S[1:-1,1:-1] = 1.5
#S[1,1] = 5.
#S[2,3] = 6.
#S[3,4] = 5.
#S[2,4] = 6.
#S[1:-1,1:-1] = np.random.uniform(low=0.0, high=1.0, size=(ny-2,nx-2))

# Infection initialization:
I = np.zeros((ny, nx))
I[1:-1,1:-1] = 0.0
#I[3,4] = 1.

# Recovery initialization:
Rc = np.zeros((ny, nx))
Rc[1:-1,1:-1] = 0.0

# Agent initialization:
Ac = np.zeros((ny, nx))
Ac[1:-1,1:-1] = 4.0

# Parameters:
Ds = 1.0e-2
Di = 1.0e-2
rho = 5.0e-2
gamma = 1.
eta = 1.0e-2
dt = 1.0e-1

# Time parameters:
S_old = np.zeros((ny, nx))
I_old = np.zeros((ny, nx))
R_old = np.zeros((ny, nx))
A_old = np.zeros((ny, nx))

nt = 10.
t = 0.

while t < nt:
    S_old[1:-1,1:-1] = S[1:-1,1:-1]
    I_old[1:-1,1:-1] = I[1:-1,1:-1]
    R_old[1:-1,1:-1] = Rc[1:-1,1:-1]
    A_old[1:-1,1:-1] = Ac[1:-1,1:-1]
    S_sol = solve_Succeptibility(Ds,I_old,S_old,A_old,dt,ny,nx)
    I_sol = solve_Infection(Di,I_old,S_old,A_old,dt,ny,nx)
    R_sol = solve_Recovery(I_old,R_old,dt,nx,ny)
    A_sol = solve_Agent(I_old,S_old,A_old,eta,gamma,rho,dt,ny,nx)
    S[1:-1,1:-1] = S_sol
    I[1:-1,1:-1] = I_sol
    Rc[1:-1,1:-1] = R_sol
    Ac[1:-1,1:-1] = A_sol
    
    t = t+dt


X = np.linspace(0., lx, nx)
Y = np.linspace(0., ly, ny)
# plotting contours
plt.figure()
e = plt.contourf(X, Y, Rc, cmap='BrBG')
plt.title('Recovery')
plt.colorbar(e)
plt.figure()
e = plt.contourf(X, Y, S, cmap='ocean_r')
plt.title('Succeptibility')
plt.colorbar(e)
plt.figure()
e = plt.contourf(X, Y, I, cmap='RdBu')
plt.title('Infection')
plt.colorbar(e)
plt.figure()
e = plt.contourf(X, Y, Ac, cmap='twilight')
plt.title('Agent')
plt.colorbar(e)

plt.show()



