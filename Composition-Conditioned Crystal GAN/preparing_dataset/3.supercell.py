import os
import glob
import numpy as np
from ase.io import read, write
from ase import Atoms,Atom
import view_atoms_mgmno
import itertools
import make_representation
from tqdm import tqdm
import pickle

def do_supercell(image, n = None):
    atoms, image = view_atoms_mgmno.view_atoms(image, view = False)
    s  = atoms.get_chemical_symbols()
    n_mg = s.count("Mg")
    n_mn = s.count('Mn')
    n_o = s.count('O')
#    print atoms

    atoms_x = atoms.repeat((2,1,1))
    atoms_y = atoms.repeat((1,2,1))
    atoms_z = atoms.repeat((1,1,2))
#    print atoms_x

#    image_origin = make_representation.do_feature(atoms).reshape(1,22,3)
    image_x = make_representation.do_feature(atoms_x).reshape(1,30,3)
    image_y = make_representation.do_feature(atoms_y).reshape(1,30,3)
    image_z = make_representation.do_feature(atoms_z).reshape(1,30,3)

#    new_images= []
    new_images = np.vstack((image_x,image_y,image_z))
    return new_images

if __name__ == "__main__":
#    images = np.load('unique_vo.npy')
    images = np.load('unique_mgmno_comp_dict',allow_pickle=True)
    with open("unique_mgmno_name_list" ,'rb') as f:
        name_list = pickle.load(f)
    result_comp_dict = {}
    k = images.keys()
    new_name_list = []
#    image = images[500]
    count = 0
    s = 0
    for comp in tqdm(k):
#        print comp
        temp = comp.split('_')
        n_mg = int(temp[0])
        n_mn = int(temp[1])
        n_o = int(temp[2])
        if n_mg <= 4 and n_mn <= 4 and n_o <= 6:
            print(comp)
            #print "original image is ", image
#    for i in range(images.shape[0]):
            image_and_name_list = images[comp]
            print(len(image_and_name_list))
            s += len(image_and_name_list)
#            for ii,image in enumerate(image_and_name_list):
            for ii in range(len(image_and_name_list)):
                name = image_and_name_list[ii][1]
                    
#                name = name_list[name_index]
                image = image_and_name_list[ii][0]
                t = do_supercell(image)
#                print t.shape
                if count == 0 and ii == 0:
                    supercell_data = t
                    new_name_list = [name,name,name]
#                    print "save_ : ", comp
                else:
                    supercell_data = np.vstack((supercell_data,t))
                    new_name_list += [name,name,name]
#                    print "save : ", comp
                print(supercell_data.shape)
                print(len(new_name_list))
            count += 1
        else:
            pass

    print("supercell data shape is ",supercell_data.shape)
    print("new name data length is ",len(new_name_list))
    print(count)
#    print new_data.shape
    np.save("supercell_only",supercell_data)               
    with open("supercell_only_name_list", 'wb') as f:
        pickle.dump(new_name_list, f)

    print(s*3)
    
#    print t.shape
#    for i in t:
#            print i
