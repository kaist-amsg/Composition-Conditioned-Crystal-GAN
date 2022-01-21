import numpy as np
from ase import Atoms
from ase.io import read,write
import sys

#temp = sys.argv[1]
#n = int(sys.argv[2])
#nn = int(sys.argv[3])

#path = './'+temp+'/gen_images_'+str(n)+'.npy'

#dat = np.load(path)

#x = dat[nn]
#x = x.reshape(-1,3)
#cell = x[:3,:]*15
#pos = x[3:,:]

def back_to_10_cell(scaled_pos,n_mg,n_mn,n_o):
    cell = np.identity(3)*15
    atoms = Atoms('Mg'+str(n_mg)+'Mn'+str(n_mn)+'O'+str(n_o))
    atoms.set_cell(cell)
    atoms.set_scaled_positions(scaled_pos)
    pos = atoms.get_positions()	

    cell = np.identity(3)*10
    pos = pos - np.array([2.5,2.5,2.5])
    atoms = Atoms('Mg'+str(n_mg)+'Mn'+str(n_mn)+'O'+str(n_o))
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    scaled_poss = atoms.get_scaled_positions()
    return scaled_poss

def back_to_real_cell(scaled_pos, real_cell, n_mg,n_mn,n_o):
    atoms = Atoms('Mg'+str(n_mg)+'Mn'+str(n_mn)+'O'+str(n_o))
    atoms.set_cell(real_cell)
    atoms.set_scaled_positions(scaled_pos)
    return atoms

def remove_zero_padding(pos):
    criteria = 0.4
    mg_pos = pos[:8,:]
    mn_pos = pos[8:16,:]
    o_pos = pos[16:,:]
#	print "V : ",v_pos
#	print "O  : ",o_pos
    mg = np.sum(mg_pos, axis=1)
    mn = np.sum(mn_pos, axis=1)
    o = np.sum(o_pos, axis=1)
    mg_index = np.where(mg > criteria)
    mn_index = np.where(mn > criteria)
    o_index = np.where(o > criteria)	
#	print v_index
#	print o_index
#	print "V : ",v_pos[v_index]	
#	print "O  : ",o_pos[o_index]
    n_mg = len(mg_index[0])
    n_mn = len(mn_index[0])
    n_o = len(o_index[0])
    mg_pos = mg_pos[mg_index]
    mn_pos = mn_pos[mn_index]
    o_pos = o_pos[o_index]
    if n_mg == 0:
        mg_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_mg = 1
    if n_mn == 0:
        mn_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_mn = 1
    if n_o == 0:
        o_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_o = 1

    pos = np.vstack((mg_pos,mn_pos,o_pos))
    return pos, n_mg,n_mn, n_o

def view_atoms(image, view = True):
#        path = './'+temp+'/gen_images_'+str(n)+'.npy'

#        dat = np.load(path)

#        x = dat[nn]
    x = image
    x = x.reshape(-1,3)
#        cell = x[:2,:]
    l = x[0,:]*30
    a = x[1,:]*180
    cell = np.hstack((l,a))
    pos=x[2:,:]

#        cell = x[:3,:]*30
#        pos = x[3:,:]
#       print pos
    pos,n_mg, n_mn,n_o = remove_zero_padding(pos)
#       print n_ca,n_mn,n_o
    scaled_pos = back_to_10_cell(pos,n_mg,n_mn,n_o)
    atoms = back_to_real_cell(scaled_pos, cell, n_mg,n_mn,n_o)
    atoms.set_pbc([1,1,1])
    if view:
        atoms.edit()
    return atoms, x
#!/usr/bin/env python
#atoms = view_atoms(temp, n , nn)

if '__name__' == '__main__':
        pass
else:
        print("import")


