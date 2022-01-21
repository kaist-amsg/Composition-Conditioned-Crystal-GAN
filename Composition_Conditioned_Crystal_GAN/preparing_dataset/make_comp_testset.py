import numpy as np
import os
import glob
from ase.io import read, write
from ase import Atoms,Atom
import itertools
#from permutation import do_permutation
#from translation import do_translation
#from permutation import rearrangement_vertical
#from supercell import do_supercell
import make_representation
import view_atoms_mgmno
from tqdm import tqdm
import pickle
from collections import Counter
import sys

def permutation(image):
    c = image[:2,:]
    mg = image[2:10,:]
    mn = image[10:18,:]
    o = image[18:,:]
    mg_l = list(range(8))
    mn_l = list(range(8))
    o_l = list(range(12))
    mg_index = np.random.choice(mg_l,8,replace=False)
    mn_index = np.random.choice(mn_l,8,replace=False)
    o_index = np.random.choice(o_l,12,replace=False)
    new_mg = mg[mg_index,:]
    new_mn = mn[mn_index,:]
    new_o = o[o_index,:]
    new_image = np.vstack((c,new_mg,new_mn,new_o))
    return new_image


def do_translation(image, n = None):
    cellcell = image[:2,:]
    atoms, image = view_atoms_mgmno.view_atoms(image, view=False)
    atoms0 = atoms.copy()
    pos = atoms0.get_positions()
    cell = atoms0.get_cell()
    image_list = []
 #   for i in range(n):
    delta = np.random.uniform(0,1,size = 3).reshape(1,3)
    delta = np.multiply(np.linalg.norm(cell, axis = 1),delta)
    new_pos = pos + delta
    atoms.set_positions(new_pos)
    new_atoms = atoms.copy()   
    new_image = make_representation.do_feature(new_atoms)
    temp = new_image[2:,:]
    final_new_image = np.vstack((cellcell,temp))
    final_new_image = permutation(final_new_image)
    return final_new_image

def remain(image_list, b):
    m = len(image_list)

#    t = np.array(image_list)
    mm = np.arange(m)
    index_list = np.random.choice(mm, b)
    remain_list = []
    name_list = []
    for index in index_list:
        image = image_list[index][0]
        name = image_list[index][1]
        image_t = do_translation(image)
        remain_list.append(image_t)
        name_list.append(name)
    return remain_list,name_list

comp_image_dict = np.load('unique_mgmno_comp_dict', allow_pickle =True)
comp_list = comp_image_dict.keys()

new_comp_image = {}
print("load data")
names = []
#ag_number = 300
ag_number = int(sys.argv[1])
for ii,comp in enumerate(comp_list):
    comp_number = int(len(comp_image_dict[comp]))
    if comp == '6_3_11':
        print("compositions is ", comp)
        print("number of images is ", comp_number)
        print("number of images is ", len(comp_image_dict[comp]))
        temp = comp.split('_')
        a = ag_number/comp_number
        b = ag_number%comp_number
        if comp_number <= ag_number/4:
            n_r = 3

        elif comp_number <= ag_number/2:
            n_r = 1

        else:
            n_r = 0
        
        
        comp_number_rot = comp_number*(n_r+1)
        n_t = int(ag_number/comp_number_rot -1)
        t_c = ag_number-(n_t+1)*comp_number_rot
        final_images = []
        final_names = []
        print("n_r is ", n_r)
        print("n_t is ", n_t)
        print("t_c is ", t_c)
        print('sum is ', comp_number * (n_r+1) * (n_t+1) + t_c)
        for i in range(len(comp_image_dict[comp])):
            image = comp_image_dict[comp][i][0]
            name = comp_image_dict[comp][i][1]
            t_c_list = []
            image_after_rotation = [image]
            name_after_rotation = [name]
            x_r = image[:,0].reshape(30,1)
            y_r = image[:,1].reshape(30,1)
            z_r = image[:,2].reshape(30,1)
            r1 = np.hstack((y_r,x_r,z_r))
            r2 = np.hstack((z_r,y_r,x_r))
            r3 = np.hstack((x_r,z_r,y_r))
            if n_r == 3:
                image_after_rotation += [r1,r2,r3]
                name_after_rotation += [name,name,name]
            elif n_r == 1:
            
                image_after_rotation += [r2]
                name_after_rotation += [name]

            elif n_r == 0:
                pass

            m = len(image_after_rotation)
            image_after_rotation = np.array(image_after_rotation)
            
            image_after_translation = []
            name_after_translation = []

            for iii in range(m):
                image_ = image_after_rotation[iii]
                image_after_translation.append(image_)
                name_after_translation.append(name)
                for iiii in range(n_t):
                    t_image = do_translation(image_)
                    image_after_translation.append(t_image)
                    name_after_translation.append(name)

            print('images after translation : ', len(image_after_translation))
            print('name after translation : ', len(name_after_translation))
        
        
            final_images = final_images + image_after_translation
            final_names = final_names + name_after_translation

        remain_list,remain_list_name = remain(comp_image_dict[comp], t_c)
        print('before remain : ',len(final_images))
        print('remain : ',len(remain_list))
        final_images = final_images + remain_list
        final_names = final_names + remain_list_name
        print('after remain : ',len(final_images))
        print('after remain name :', len(final_names))
        final+= final_images
        names+= final_names
final = np.array(final)
print(final.shape)
np.save('mgmno_6311_'+str(ag_number),final)
with open("mgmno_6311_names_"+str(ag_number), 'wb') as f:
    pickle.dump(names, f)

