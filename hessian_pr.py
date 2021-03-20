import numpy as np
import re
import json
from scipy.linalg import *
from sympy import *
import sys

elements = ['H', 'He',
         'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
         'Na', 'Mg','Al', 'Si', 'P', 'S', 'Cl', 'Ar',
         'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
         'Rb', 'Sr','Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
         'Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
         'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
         'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Uub']

#Units
Hartree = 2625500.2 # J/mol
Bohr = 5.29177210903E-11 #m
me = 9.1093837015E-31 #Kg
#me = 1.6605390420E-27
c = 29979245800 #m/s
NA =  6.02214076E23 #

def read_xyz(xyz):
    atoms = list()
    x = list()
    y = list()
    z = list()

    with open(xyz) as fp:
        # Skip the first two lines.
        next(fp)
        next(fp)
        for line in fp:
            data = line.split()
            atoms.append(data[0])
            x.append(float(data[1]))
            y.append(float(data[2]))
            z.append(float(data[3]))

    return atoms, np.array(x), np.array(y), np.array(z)

def read_hessian(hessian_file):
    #Open and 'read' the file
    hess = open(hessian_file)
    hess = hess.readlines()
    #Create a list with all the numerical values contained in the file
    hess_data_values = [float(value) for line in hess for value in line.split()]
    #Get the number of atoms
    natoms = int(hess_data_values[0])
    #and delete from the list
    hess_data_values = np.array([hess_data_values[i] for i in range(1,len(hess_data_values))])
    ##Hessian matrix 3N x 3N
    hessian = hess_data_values.reshape((3*natoms,3*natoms))

    return(hessian,natoms)


def mass_weighted_hessian(hessian_file,xyz_file):
    atoms,x_coords,y_coords,z_coords = read_xyz(xyz_file)
    hessian,natoms = read_hessian(hessian_file)
    print(np.shape(hessian))
    print(atoms)
      
    with open('MASSES.json') as json_file:
        mass_dat = json.load(json_file)

    #Obtain the masses of the atoms in the analyzed molecule
    masses = [mass_dat[str(elements.index(atom) + 1)] for atom in atoms]
    #masses = [12.0107,12.0107,12.0107,12.0107,12.0107,12.0107,1.0008,1.0008,1.0008,1.0008,1.0008,1.0008]
    #masses = [15.99903,1.00811,1.00811]
    mw_hessian = np.zeros((3*natoms,3*natoms))
    mw_test = np.zeros((3*natoms,3*natoms))
    k = 0
    for i in range(0,natoms*3):
        for j in range(0,natoms):
            mw_hessian[i,3*j] = hessian[i,3*j] * (1/np.sqrt(masses[k]*masses[j]))
            mw_hessian[i,3*j+1] = hessian[i,3*j+1] * (1/np.sqrt(masses[k]*masses[j]))
            mw_hessian[i,3*j+2] = hessian[i,3*j+2] * (1/np.sqrt(masses[k]*masses[j]))
            mw_test[i,3*j] = str(k + 1) + str(j + 1)
            mw_test[i,3*j+1] = str(k + 1) + str(j + 1)
            mw_test[i,3*j+2] = str(k +1) + str(j + 1)
        if (i + 1)%3 == 0:
            k += 1
    print(mw_test)
    return(mw_hessian)

#mw_hessian = mass_weighted_hessian('h2o_hessian.dat','h2o.xyz')
#mw_hessian = mass_weighted_hessian('benzene_hessian.dat','benzene.xyz')
mw_hessian = mass_weighted_hessian('3c1b_hessian.dat','3c1b.xyz')


#Diagonalize the matrix. Sympy diagonalize returns a tuple (P,D)
# M = PxDxP^-1
M = Matrix(mw_hessian)
P, D = M.diagonalize()
dd = np.diag(D)
#The eigenvalues of the mwh are in units of Hartree/me * Bohr^2
dd = [round(d,10) for d in dd]
dd.sort(reverse=True)
#print(dd)

#transform units of diagonal elements to 1/s**2
units = Hartree/(NA*me*(Bohr*Bohr))
du = [d*units for d in dd]

#vibrational frequencies are proportional to the sqrt of the evals of the mwh:
vibs = [sqrt(u/(4*(np.pi**2)*(c**2))) for u in du]
print(vibs)