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
#sc_comp_dict = {'4_4_8': 389, '4_4_12': 351, '2_2_6': 199, '2_2_4': 193, '4_2_8': 133, '2_4_8': 131, '4_8_12': 117, '8_4_12': 111, '4_4_10': 93, '4_2_6': 70, '2_4_6': 69, '2_6_12': 53, '6_2_10': 51, '2_6_8': 47, '2_6_10': 44, '6_2_8': 44, '2_4_10': 40, '1_1_3': 40, '1_1_2': 32, '1_4_8': 23, '1_2_4': 20, '2_1_4': 20, '2_2_5': 16, '4_6_12': 15, '6_4_12': 15, '6_4_10': 13, '1_3_6': 12, '2_3_8': 11, '2_8_10': 11, '2_1_3': 11, '1_3_4': 11, '4_3_8': 10, '1_2_3': 10, '8_2_10': 10, '4_6_10': 10, '3_3_9': 10, '3_1_4': 10, '8_2_12': 9, '2_8_12': 9, '3_4_8': 9, '1_6_11': 8, '1_2_5': 7, '3_3_8': 7, '6_3_9': 6, '1_4_7': 6, '2_4_7': 6, '6_6_12': 6, '2_3_6': 5, '4_2_7': 5, '3_2_6': 5, '3_5_12': 5, '3_5_10': 5, '3_6_9': 5, '5_4_10': 4, '1_4_9': 4, '4_5_10': 4, '3_2_5': 4, '2_5_10': 4, '1_5_8': 4, '5_3_10': 4, '6_1_8': 3, '4_1_6': 3, '2_5_12': 3, '1_7_12': 3, '1_6_8': 3, '5_2_8': 3, '7_1_8': 3, '2_5_8': 3, '2_3_5': 3, '1_4_6': 3, '1_7_8': 3, '1_5_9': 3, '5_5_10': 2, '1_5_10': 2, '3_5_8': 2, '6_2_9': 2, '5_1_6': 2, '4_4_11': 2, '4_3_9': 2, '1_7_11': 2, '3_3_6': 2, '3_1_5': 2, '1_6_12': 2, '2_4_9': 2, '1_5_6': 2, '1_3_7': 2, '1_3_5': 2, '5_3_8': 2, '2_6_9': 2, '3_6_11': 1, '2_6_11': 1, '5_2_7': 1, '6_3_11': 1, '5_4_9': 1, '1_5_11': 1, '4_5_9': 1, '3_5_9': 1, '4_1_5': 1, '5_3_9': 1, '2_5_7': 1, '1_4_5': 1, '1_8_11': 1, '8_1_9': 1, '3_5_11': 1, '7_2_9': 1, '2_7_9': 1, '3_4_9': 1, '5_3_11': 1, '1_5_7': 1, '3_4_11': 1, '1_8_9': 1, '5_1_7': 1}

#sc_comp_dict = {'4_4_8': 406, '4_4_12': 365, '2_2_6': 178, '2_2_4': 171, '4_2_8': 131, '2_4_8': 129, '4_8_12': 117, '8_4_12': 114, '4_4_10': 93, '2_4_6': 69, '4_2_6': 68, '2_6_12': 53, '6_2_10': 51, '2_6_10': 44, '2_6_8': 41, '2_4_10': 40, '6_2_8': 40, '1_1_3': 32, '1_1_2': 23, '1_4_8': 23, '1_2_4': 19, '2_1_4': 19, '2_2_5': 16, '4_6_12': 15, '6_4_12': 15, '6_4_10': 13, '1_3_6': 12, '6_6_12': 12, '2_8_10': 11, '2_3_8': 11, '1_2_3': 10, '2_1_3': 10, '3_3_9': 10, '8_2_10': 10, '4_3_8': 10, '4_6_10': 10, '3_4_8': 9, '2_8_12': 9, '8_2_12': 9, '1_3_4': 8, '3_1_4': 8, '1_6_11': 8, '1_2_5': 7, '3_3_8': 7, '2_4_7': 6, '1_4_7': 6, '6_3_9': 6, '2_3_6': 5, '3_2_6': 5, '3_5_10': 5, '4_2_7': 5, '3_5_12': 5, '3_6_9': 5, '3_3_6': 4, '5_3_10': 4, '2_5_10': 4, '1_5_8': 4, '3_2_5': 4, '1_4_9': 4, '4_5_10': 4, '5_4_10': 4, '2_5_12': 3, '5_2_8': 3, '2_5_8': 3, '1_7_12': 3, '6_1_8': 3, '1_6_8': 3, '7_1_8': 3, '1_7_8': 3, '1_4_6': 3, '4_1_6': 3, '2_3_5': 3, '1_5_9': 3, '1_6_12': 2, '5_1_6': 2, '1_5_6': 2, '4_3_9': 2, '6_2_9': 2, '2_6_9': 2, '1_3_7': 2, '4_4_11': 2, '1_7_11': 2, '1_3_5': 2, '3_1_5': 2, '5_3_8': 2, '3_5_8': 2, '2_4_9': 2, '1_5_10': 2, '5_5_10': 2, '3_4_9': 1, '5_3_9': 1, '3_5_9': 1, '3_6_11': 1, '4_1_5': 1, '1_4_5': 1, '6_3_11': 1, '2_6_11': 1, '1_8_11': 1, '5_3_11': 1, '3_5_11': 1, '1_5_11': 1, '3_4_11': 1, '2_7_9': 1, '7_2_9': 1, '1_5_7': 1, '5_1_7': 1, '5_4_9': 1, '4_5_9': 1, '2_5_7': 1, '5_2_7': 1, '8_1_9': 1, '1_8_9': 1}
#comp_dict = {'4_4_12': 114, '4_4_8': 98, '2_2_4': 97, '2_2_6': 79, '4_2_8': 73, '2_4_8': 71, '6_2_10': 45, '4_4_10': 45, '1_1_3': 40, '2_4_6': 39, '2_6_10': 38, '4_2_6': 37, '1_1_2': 32, '1_4_8': 23, '1_2_4': 20, '2_1_4': 20, '2_4_10': 19, '2_6_12': 17, '2_2_5': 16, '6_2_8': 14, '2_6_8': 14, '1_3_6': 12, '1_3_4': 11, '2_1_3': 11, '2_3_8': 11, '3_1_4': 10, '1_2_3': 10, '3_3_9': 10, '4_3_8': 10, '3_4_8': 9, '2_8_10': 8, '1_6_11': 8, '1_2_5': 7, '8_2_10': 7, '3_3_8': 7, '2_4_7': 6, '1_4_7': 6, '6_3_9': 6, '2_3_6': 5, '3_2_6': 5, '3_5_10': 5, '4_2_7': 5, '3_5_12': 5, '3_6_9': 5, '5_3_10': 4, '2_5_10': 4, '1_5_8': 4, '3_2_5': 4, '1_4_9': 4, '4_5_10': 4, '5_4_10': 4, '2_5_12': 3, '5_2_8': 3, '2_5_8': 3, '1_7_12': 3, '6_1_8': 3, '1_6_8': 3, '7_1_8': 3, '1_7_8': 3, '1_4_6': 3, '4_1_6': 3, '2_3_5': 3, '1_5_9': 3, '1_6_12': 2, '5_1_6': 2, '1_5_6': 2, '4_3_9': 2, '6_2_9': 2, '2_6_9': 2, '1_3_7': 2, '3_3_6': 2, '4_4_11': 2, '1_7_11': 2, '1_3_5': 2, '3_1_5': 2, '5_3_8': 2, '3_5_8': 2, '2_4_9': 2, '1_5_10': 2, '5_5_10': 2, '3_4_9': 1, '5_3_9': 1, '3_5_9': 1, '3_6_11': 1, '4_1_5': 1, '1_4_5': 1, '6_3_11': 1, '2_6_11': 1, '1_8_11': 1, '5_3_11': 1, '3_5_11': 1, '1_5_11': 1, '3_4_11': 1, '2_7_9': 1, '7_2_9': 1, '6_4_10': 1, '4_6_10': 1, '1_5_7': 1, '5_1_7': 1, '5_4_9': 1, '4_5_9': 1, '2_5_7': 1, '5_2_7': 1, '8_1_9': 1, '1_8_9': 1} 

#comp_dict = {'4_4_12': 119, '2_2_4': 102, '4_4_8': 100, '2_2_6': 82, '4_2_8': 74, '2_4_8': 72, '6_2_10': 45, '4_4_10': 45, '2_4_6': 39, '4_2_6': 38, '2_6_10': 38, '1_1_3': 32, '1_1_2': 23, '1_4_8': 23, '1_2_4': 19, '2_1_4': 19, '2_4_10': 19, '2_6_8': 17, '2_6_12': 17, '6_2_8': 16, '2_2_5': 16, '1_3_6': 12, '2_3_8': 11, '1_2_3': 10, '2_1_3': 10, '3_3_9': 10, '4_3_8': 10, '3_4_8': 9, '1_3_4': 8, '3_1_4': 8, '2_8_10': 8, '1_6_11': 8, '1_2_5': 7, '8_2_10': 7, '3_3_8': 7, '2_4_7': 6, '1_4_7': 6, '6_3_9': 6, '2_3_6': 5, '3_2_6': 5, '3_5_10': 5, '4_2_7': 5, '3_5_12': 5, '3_6_9': 5, '3_3_6': 4, '5_3_10': 4, '2_5_10': 4, '1_5_8': 4, '3_2_5': 4, '1_4_9': 4, '4_5_10': 4, '5_4_10': 4, '2_5_12': 3, '5_2_8': 3, '2_5_8': 3, '1_7_12': 3, '6_1_8': 3, '1_6_8': 3, '7_1_8': 3, '1_7_8': 3, '1_4_6': 3, '4_1_6': 3, '2_3_5': 3, '1_5_9': 3, '1_6_12': 2, '5_1_6': 2, '1_5_6': 2, '4_3_9': 2, '6_2_9': 2, '2_6_9': 2, '1_3_7': 2, '4_4_11': 2, '1_7_11': 2, '1_3_5': 2, '3_1_5': 2, '5_3_8': 2, '3_5_8': 2, '2_4_9': 2, '1_5_10': 2, '5_5_10': 2, '3_4_9': 1, '5_3_9': 1, '3_5_9': 1, '3_6_11': 1, '4_1_5': 1, '1_4_5': 1, '6_3_11': 1, '2_6_11': 1, '1_8_11': 1, '5_3_11': 1, '3_5_11': 1, '1_5_11': 1, '3_4_11': 1, '2_7_9': 1, '7_2_9': 1, '6_4_10': 1, '4_6_10': 1, '1_5_7': 1, '5_1_7': 1, '5_4_9': 1, '4_5_9': 1, '2_5_7': 1, '5_2_7': 1, '8_1_9': 1, '1_8_9': 1}

#with open("unique_mgmno_sc_comp_dict", 'rb') as f:
 #   comp_image_dict = pickle.load(f)
#with open("unique_mgmno_comp_dict", 'rb') as f:
#    comp_image_dict = pickle.load(f)

comp_image_dict = np.load('unique_sc_mgmno_comp_dict', allow_pickle =True)
#comp_image_dict = np.load("unique_mgmno_comp_dict", allow_pickle = True)
#comp_list = comp_image_dict.keys()
comp_list = comp_image_dict.keys()

new_comp_image = {}
print("load data")
#print(len(comp_dict.keys()))
final = []
names = []
#ag_number = 300
ag_number = int(sys.argv[1])
for ii,comp in enumerate(comp_list):
#    comp_number = int(sc_comp_dict[comp])
#    comp_number = int(comp_dict[comp])
    comp_number = int(len(comp_image_dict[comp]))
    print("compositions is ", comp)
    print("number of images is ", comp_number)
    print("number of images is ", len(comp_image_dict[comp]))
    temp = comp.split('_')
#    n_v = int(temp[0])
#    n_o = int(temp[1])
    # (n_p + 1) * (n_t + 1) = image_number
    a = ag_number/comp_number
    b = ag_number%comp_number
    if comp_number <= ag_number/4:
#        n_p = a/10 -1
        n_r = 3
#        comp_number_rot = comp_number*(n_r+1)
#        n_t = int(ag_number/comp_number_rot -1)
#        n_t = a/(n_p+1) -1
#        t_c = ag_number- (n_t+1)*comp_number_rot

    elif comp_number <= ag_number/2:
        n_r = 1

    else:
        n_r = 0
        
        
    comp_number_rot = comp_number*(n_r+1)
    n_t = int(ag_number/comp_number_rot -1)
#        n_t = a - 1
    t_c = ag_number-(n_t+1)*comp_number_rot
#    n_p = 5
#    n_t = 3
    final_images = []
    final_names = []
    print("n_r is ", n_r)
    print("n_t is ", n_t)
    print("t_c is ", t_c)
    print('sum is ', comp_number * (n_r+1) * (n_t+1) + t_c)
#    for i,image in enumerate(comp_image_dict[comp]):
    for i in range(len(comp_image_dict[comp])):
        image = comp_image_dict[comp][i][0]
        name = comp_image_dict[comp][i][1]
        t_c_list = []
#        print image
#        print image.shape
        #if not n_p == 0 :
        #    image_after_permutation = [image]
        image_after_rotation = [image]
        name_after_rotation = [name]
        #print image
#        if not n_p == 0:
#            for ii in range(n_p):
#                p_image = permutation(image)
        #    print p_image
#                image_after_permutation.append(p_image)
#           t_c_list= []
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
#            print image_
#            print image_.shape
            image_after_translation.append(image_)
            name_after_translation.append(name)
            for iiii in range(n_t):
                t_image = do_translation(image_)
                image_after_translation.append(t_image)
                name_after_translation.append(name)

        print('images after translation : ', len(image_after_translation))
        print('name after translation : ', len(name_after_translation))
#        if not len(t_c_list) == 0:
#            image_after_translation += t_c_list

#        print 'images after summation of t_c_list :', len(image_after_translation) 
        
        
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
#    if i == 7:
#        break
final = np.array(final)
print(final.shape)
np.save('mgmno_'+str(ag_number),final)
with open("mgmno_names_"+str(ag_number), 'wb') as f:
    pickle.dump(names, f)
#    if i == 7:
#        break        
#print len(final_images)

