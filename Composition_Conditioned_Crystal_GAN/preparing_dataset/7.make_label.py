import numpy as np
from ase.io import read,write
from ase import Atoms

from view_atoms_mgmno import *
import pickle
from tqdm import tqdm



def check(image):
	pos = image[2:,:]
	mg = pos[:8,:]
	mn = pos[8:16,:]
	o = pos[16:,:]
	mgmg = np.sum(mg,axis=1)
	mgmgmg = np.zeros((8,1)) + 1
	mgmgmg[mgmg<0.4] = 0
	mnmn = np.sum(mn,axis=1)
	mnmnmn = np.zeros((8,1)) +1
	mnmnmn[mnmn<0.4] = 0
	oo = np.sum(o,axis=1)
	ooo = np.zeros((12,1)) + 1
	ooo[oo<0.4] = 0 
	label = np.vstack((mgmgmg,mnmnmn,ooo))
	print(label.shape)
	return label




a = np.load("mgmno_100.npy")

m = a.shape[0]
output = []
for i in tqdm(range(m)):
	x = a[i]
#	print x
#	atoms,x = view_atoms(x,view=False)
#	print atoms
#	s = atoms.get_chemical_symbols()
#	n_v = s.count('V')
#	n_o = s.count('O')
#	new_input = (x,np.array([n_v,n_o]))
	label = check(x)
	print(x)
	print(label)
	new_input = (x,label)


	output.append(new_input)

with open('mgmno_100.pickle', 'wb') as f:
	pickle.dump(output,f)


