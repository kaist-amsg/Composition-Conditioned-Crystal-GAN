import glob
import pickle
import numpy as np




with open("supercell_only_name_list",'rb') as f:
    sc_name_list = pickle.load(f)
with open("unique_mgmno_name_list",'rb') as f:
    name_list = pickle.load(f)

sc = np.load("supercell_only.npy" )
images = np.load("unique_mgmno.npy")


print(sc.shape)
print(images.shape)

print(len(sc_name_list))
print(len(name_list))


result = np.vstack((images,sc))
result_name = name_list+sc_name_list

print(result.shape)
print(len(result_name))


np.save('unique_sc_mgmno' , result)
with open("unique_sc_mgmno_name_list", 'wb') as f:
    pickle.dump(result_name, f)

