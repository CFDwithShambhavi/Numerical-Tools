# Author: Shambhavi Nandan

# This code implements 2D diffusion model for pandemic spread. The original paper can be found her: https://api.semanticscholar.org/CorpusID:9108055.oa

import numpy as np
import CSV_FileReader_Writer as csv
import MatrixTools as mt
import os


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
    
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
    sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
    
    sol = np.reshape(sol,(ny-2,nx-2))
    
    return sol

def solve_Infection(Di,Ic_old,Sc_old,Ac_old,dt,ny,nx):
    
    n_it = 0 # iterator

    e = 1.0e-3 # tolerance for convergence
    
    Ic_guess = np.zeros((ny,nx))
    Ic_guess[:,:] = Ic_old.copy()
    
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
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Di/dx**2
                        at[j,i] = 0.
                        ab[j,i] = Di/dx**2
                        source[i,j] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # boundary nodes:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = Di/dx**2
                        at[j,i] = 0.
                        ab[j,i] = Di/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i==imax: # right corner node:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = 0.
                        at[j,i] = 0.
                        ab[j,i] = Di/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
    
            elif j==jmax: # bottom row
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left corner node:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Di/dx**2
                        at[j,i] = Di/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # boundary nodes:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = Di/dx**2
                        at[j,i] = Di/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i==imax: # right corner node:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = 0.
                        at[j,i] = Di/dx**2
                        ab[j,i] = 0.
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
    
            else: # internal rows
                imin = 1
                imax = nx-2
                for i in np.arange(imin, imax+1):
                    if i==imin: # left boundary node:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = 0.
                        ar[j,i] = Di/dx**2
                        at[j,i] = Di/dx**2
                        ab[j,i] = Di/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
                    elif i>imin and i<imax: # internal nodes:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = Di/dx**2
                        at[j,i] = Di/dx**2
                        ab[j,i] = Di/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt 
                    elif i==imax: # right boundary node:
                        ac[j,i] = (1./dt) + ss + 4*Di/dx**2
                        al[j,i] = Di/dx**2
                        ar[j,i] = 0.
                        at[j,i] = Di/dx**2
                        ab[j,i] = Di/dx**2
                        source[j,i] = Sc_old[j,i]*(Ic_guess[j,i]+Ac_old[j,i]) + Ic_old[j,i]/dt
        
        A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
        sol = np.linalg.solve(A,source[1:-1,1:-1].flatten())
        
        sol = np.reshape(sol,(ny-2,nx-2))
        
        print np.sum(np.abs(Ic_guess[1:-1,1:-1]-sol))/((ny-2)*(nx-2)),'\n'
        
        if np.all(np.abs(Ic_guess[1:-1,1:-1]-sol) <= e) or n_it == 100:
            print np.sum(np.abs(Ic_guess[1:-1,1:-1]-sol))/((ny-2)*(nx-2)),'\n'
            print'******** Convergence for Infection ********','\n'
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
    
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
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
    
    A = mt.assemble_a_to_A(ny-2, nx-2, ac[1:-1,1:-1],ar[1:-1,1:-1],al[1:-1,1:-1],at[1:-1,1:-1],ab[1:-1,1:-1])
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
S = np.zeros((ny, nx))
#S[4,4] = 3.
#S[3,4] = 3.
#S[4,3] = 3.
#S[4,5] = 3.
#S[5,4] = 3.
#S[1,1] = 5.
#S[2,3] = 6.
#S[3,4] = 5.
#S[2,4] = 6.
#S[1:-1,1:-1] = np.random.uniform(low=0.0, high=1.0, size=(ny-2,nx-2))

# Infection initialization:
I = np.zeros((ny, nx))
I[1:-1,1:-1] = 1.0
I[1,1] = 1.
#I[3,4] = 0.
#I[4,3] = 0.
#I[4,5] = 0.
#I[5,4] = 0.

# Recovery initialization:
Rc = np.zeros((ny, nx))
Rc[1:-1,1:-1] = 0.0

# Agent initialization:
Ac = np.zeros((ny, nx))
#Ac[4,4] = 4.
#Ac[3,4] = 4.
#Ac[4,3] = 4.
#Ac[4,5] = 4.
#Ac[5,4] = 4.

# Parameters:
Ds = 1.0e-2
Di = 1.0e1
rho = 5.0e-2
gamma = 1.
eta = 1.0e-2

# Time parameters:
S_old = np.zeros((ny, nx))
I_old = np.zeros((ny, nx))
R_old = np.zeros((ny, nx))
A_old = np.zeros((ny, nx))

dt = 1.0e-1
nt = 50.
t = 0.

# File save parameters
file_name = '/covidData'
path_name = os.getcwd()
save_t = 10.
    
while t <= nt:
    #print(np.abs(t-save_t),'\n')
    if np.abs(t-save_t)<=0.1 or t==0:
        print'############################ T = ',int(t)+1,' sec. ############################','\n'
        data = {}
        data['S'] = S.flatten()
        data['I'] = I.flatten()
        data['R'] = Rc.flatten()
        data['A'] = Ac.flatten()
        csv.csv_fileWriter(path_name, file_name+str(int(t)+1)+'.csv', ',', data)
        if t>0:
            save_t += 10.
            
    S_old[1:-1,1:-1] = S[1:-1,1:-1]
    I_old[1:-1,1:-1] = I[1:-1,1:-1]
    R_old[1:-1,1:-1] = Rc[1:-1,1:-1]
    A_old[1:-1,1:-1] = Ac[1:-1,1:-1]
    #S_sol = solve_Succeptibility(Ds,I_old,S_old,A_old,dt,ny,nx)
    I_sol = solve_Infection(Di,I_old,S_old,A_old,dt,ny,nx)
    #R_sol = solve_Recovery(I_old,R_old,dt,nx,ny)
    #A_sol = solve_Agent(I_old,S_old,A_old,eta,gamma,rho,dt,ny,nx)
    #S[1:-1,1:-1] = S_sol
    I[1:-1,1:-1] = I_sol
    #Rc[1:-1,1:-1] = R_sol
    #Ac[1:-1,1:-1] = A_sol
    
    t = t+dt
        





