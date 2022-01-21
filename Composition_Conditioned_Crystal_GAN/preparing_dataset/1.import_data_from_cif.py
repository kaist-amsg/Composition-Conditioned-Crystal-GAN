import glob
from ase.io import read,write
from ase import Atoms,Atom
from collections import Counter
import numpy as np
from tqdm import tqdm
from make_representation import *
import pickle

cif_path = './cifs'
vasp_path = './vasps'
cif_list = glob.glob(cif_path+'/*.cif')
vasp_list = glob.glob(vasp_path+'/*.vasp')


results = []
name_list = [] 

with open(form_e_path,'r') as f:
    lines = f.readlines()
    

for i in lines:
    temp = i.strip().split(',')

#for i in cif_list:
for i in vasp_list:
    name = i.split(',')[0]
    atoms =read(i)
    s = atoms.get_chemical_symbols()
    n_mg = s.count('Mg')
    n_mn = s.count('Mn')
    n_o = s.count('O')
    print(atoms)
    if (n_mg == 9) or (n_mn == 9):
        continue
    else:
        image = do_feature(atoms)
        print(image)

        results.append(image)
        name_list.append(name)

results = np.array(results)
print(results.shape)
np.save('unique_mgmno',results)

with open("unique_mgmno_name_list",'wb') as f:
    pickle.dump(name_list,f)

