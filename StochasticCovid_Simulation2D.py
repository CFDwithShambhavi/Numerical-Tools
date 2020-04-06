
# Author: Shambhavi Nandan

import numpy as np
import CSV_FileReader_Writer as csv
import random_neighbour as rn
from decimal import Decimal
import os

np.random.seed(19680801)

# Grid parameters:
ny=80
nx=80
ly=100
lx=100
dx=lx/nx
dy=ly/ny

# Disease parameters:
xc=0.22 # potency of infection
disease_state = ['S','I','Im']
sample2D_char = np.random.choice(disease_state, (ny,nx), p=[0.9,0.0009,0.0991]) # 2D grid filled with random disease states 'S' and 'I'
sample2D_char
# 2D grid filled with random disease states such that 0 corresponds to :'S' and 1 corresponds to :'I'
sample2D_num = np.zeros((ny,nx)) 
for index in np.ndindex(ny, nx):
    if sample2D_char[index]=='S':
        sample2D_num[index]=0
    elif sample2D_char[index]=='I':
        sample2D_num[index]=2
    else:
        sample2D_num[index]=1
#......................................................

# File parameter:
file_name = '/epi'
path_name = os.getcwd()+'/data_files'
nbSaves = 1000
nEnd = 1e6
nWrite = nEnd/nbSaves
n=0
# Writing at n=0
csv.csv_fileWriter(path_name, file_name+str(n)+'.csv', ',', {'Disease State':sample2D_num.flatten()})
#csv.csv_fileWriter(path_name, file_name+'.csv', ',', {'Disease State':sample2D_num.flatten()})

# Generating a list of indices for 2D grid of dimensions (ny,nx):
indices=[]
for i in np.ndindex(ny-1,nx-1):
    indices.append((i[0],i[1]))

Icount=0 #to count no. of infections
highI=False #indicating the state of highest infection which is set here to be 60%
# Performing stochastic analysis:
while np.any(sample2D_char=='I'):
    
    n += 1
    rand_i = indices[np.random.randint(0,len(indices)-1)]
    
    if sample2D_char[rand_i]=='I':
        c = np.random.uniform(0.,1.)
        
        if c > xc:
            rand_neig_i = rn.neighbour(rand_i,ny-1,nx-1)
            if sample2D_char[rand_neig_i]=='S':
                sample2D_char[rand_neig_i]='I'
                sample2D_num[rand_neig_i]=2
        else:
            for index in np.ndindex(ny, nx):
                if sample2D_char[index]=='I':
                    Icount += 1
            #print (Icount/(ny*nx), '\n')
            if Icount/(ny*nx) >= 0.6 and highI==False: # i.e. 60% of total population is being infected.
                print ('######### Recov ########','\n')
                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
                sample2D_num[rand_i]=1 # 2 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
                highI=True #indicating highest infection has been reached
            elif Icount/(ny*nx) <= 0.6 and Icount/(ny*nx) > 0 and highI==True:
                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
                sample2D_num[rand_i]=1 # 2 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
            elif Icount/(ny*nx) > 0.6 and highI==True:
                sample2D_char[rand_i]='R' # recovery starts after 60% of population is being infected.
                sample2D_num[rand_i]=1 # 2 represents the number equivalent of recovery, like 0 for 'S' and 1 for 'I'.
                
            Icount = 0
                
            
            
    if np.abs(n - nWrite)==0:
        csv.csv_fileWriter(path_name, file_name+str(n/2000)+'.csv', ',', {'Disease State':sample2D_num.flatten()})
        #csv.csv_fileWriter(path_name, file_name+'.csv', ',', {'Disease State':sample2D_num.flatten()})
        nWrite += nEnd/nbSaves

    if n >= nEnd:
        break



