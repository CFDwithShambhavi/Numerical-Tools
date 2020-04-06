
# Author: Shambhavi Nandan

import numpy as np
import CSV_FileReader_Writer as csv
import random_neighbour as rn
import os

# This model is inspired from SIR model given in epidemic models on lattices: https://en.wikipedia.org/wiki/Epidemic_models_on_lattices 
###############................................................................
# Explanation of Susceptible Immune Infected Rmoved (SIIR) stochastic model:
# In the "SIIR" model, there are three states:
#
# Susceptible (S) -- has not yet been infected, and has no immunity (large in number)
# Immune (I) -- are immune to the infection (very few percent)
# Infected (I)-- Already infected
# Removed (R) -- the ones recovered after treatment or died of illness
#
# Algorithm:
#
# Pick a cell in square grid or lattice
# If: it is I, then generate a random number x in (0,1).
            # If x > xc: then let I go to R. c: is the probability of an infected person to infect others.
            # elif c < xc1 and c > xc2: then I goes to Recovered state
            # esle: person with I state dies
# else: pick one nearest neighbor randomly. If the neighboring site is S, then let it become I.
# Repeat as long as there are S sites available.
###############................................................................

np.random.seed(19680801)

# Grid or lattice parameters:
ny=80
nx=80
ly=100
lx=100
dx=lx/nx
dy=ly/ny

# Disease parameters:
xc1=0.22 # threshold of potency of a person to spread infection
xc2=xc1/2. # threshold of potency of a person to recover
disease_state = ['S','I','Im']
sample2D_char = np.random.choice(disease_state, (ny,nx), p=[0.99,0.008,0.002]) # 2D grid filled with random disease states 'S' and 'I'
sample2D_char

# 2D grid filled with random disease states such that 0 corresponds to :'S' and 2 corresponds to :'I'
sample2D_num = np.zeros((ny,nx)) 
for index in np.ndindex(ny, nx):
    if sample2D_char[index]=='S': # Suceptible to infection
        sample2D_num[index]=-1
    elif sample2D_char[index]=='I': # Infected
        sample2D_num[index]=2
    else:
        sample2D_num[index]=1 # Immune or recovered
#......................................................

# File parameter:
file_name = '/epi'
path_name = os.getcwd()+'/data_files'
nbSaves = 1000 # no. of savings for data
nEnd = 1e6 # To avoid INFINTE LOOP
nWrite = nEnd/nbSaves
n=0

# Writing data at n=0
csv.csv_fileWriter(path_name, file_name+str(n)+'.csv', ',', {'Disease State':sample2D_num.flatten()})
#csv.csv_fileWriter(path_name, file_name+'.csv', ',', {'Disease State':sample2D_num.flatten()})

# Generating a list of indices for 2D grid of dimensions (ny,nx):
indices=[]
for i in np.ndindex(ny-1,nx-1):
    indices.append((i[0],i[1]))

# Ignore .....................................
#Icount=0 #to count no. of infections
#highI=False #indicating the state of highest infection which is set here to be 60%
# Ignore .....................................

# Performing stochastic analysis:
while np.any(sample2D_char=='S'): # S is more appropriate when random numbers are very close to actual random numbers
                                  # Otherwise having S here may set INFINITE LOOP.
    n += 1
    rand_i = indices[np.random.randint(0,len(indices)-1)]
    
    if sample2D_char[rand_i]=='I':
        c = np.random.uniform(0.,1.)
        
        if c > xc1:
            rand_neig_i = rn.neighbour(rand_i,ny-1,nx-1)
            if sample2D_char[rand_neig_i]=='S':
                sample2D_char[rand_neig_i]='I'
                sample2D_num[rand_neig_i]=2
        elif c < xc1 and c > xc2:
            sample2D_char[rand_i]='Rec' # recovered
            sample2D_num[rand_i]=1
            
            # Ignore .....................................
#            for index in np.ndindex(ny, nx):
#                if sample2D_char[index]=='I':
#                    Icount += 1
#            #print (Icount/(ny*nx), '\n')
#            if Icount/(ny*nx) >= 0.6 and highI==False: # i.e. 60% of total population is being infected.
#                print ('######### Recovery ########','\n')
#                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
#                sample2D_num[rand_i]=1 # 1 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
#                highI=True #indicating highest infection has been reached
#            elif Icount/(ny*nx) <= 0.6 and Icount/(ny*nx) > 0 and highI==True:
#                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
#                sample2D_num[rand_i]=1 # 1 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
#            elif Icount/(ny*nx) > 0.6 and highI==True:
#                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
#                sample2D_num[rand_i]=1 # 1 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
#                
#            Icount = 0
            # Ignore .....................................
        else:
            sample2D_char[rand_i]='Rem' # removed or died of infection
            sample2D_num[rand_i]=0
                
            
            
    if np.abs(n - nWrite)==0:
        #print(n,'\n')
        csv.csv_fileWriter(path_name, file_name+str(n/2000)+'.csv', ',', {'Disease State':sample2D_num.flatten()})
        #csv.csv_fileWriter(path_name, file_name+'.csv', ',', {'Disease State':sample2D_num.flatten()})
        nWrite += nEnd/nbSaves

    if n >= nEnd: # To avoid INFINTE LOOP
        break



