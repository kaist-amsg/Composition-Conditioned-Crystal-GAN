import argparse
import os
import numpy as np
import math
import itertools
from ase.io import read, write
from ase import Atoms, Atom
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import autograd
import copy
from view_atoms_mgmno import *
import torch.nn.init as init
from models import *
import random


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def weights_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d : 
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)
        #print( m, "initialized")


def noising(imgs):
    imgs = imgs.numpy()
    B = imgs.shape[0]
    mask = (imgs<0.01)
    a = np.random.normal(10**-3,10**-2.5,(B,1,30,3))
    noise = mask*abs(a)
    imgs_after_noising = imgs + noise
    imgs_after_noising = torch.tensor(imgs_after_noising)
    return imgs_after_noising	


def count_element(label):
    n_x  = (label==1).sum(dim=1)
    return n_x


def get_onehot(x, num_class_v, num_class_o):
    m = x.shape[0]
    output = []
    output2 = []
    for i in range(m):
        x_i = x[i]
        temp = np.zeros((num_class_v,))
        temp2 = np.zeros((num_class_o,))
        temp[x_i[0]-1] = 1
        temp2[x_i[1]-1] = 1
        output.append(temp)
        output2.append(temp2)
    output = np.array(output)
    output2 = np.array(output2)
    return output, output2

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.

    return Variable(FloatTensor(y_cat))

def make_fake_label(fake_c_mg_int,fake_c_mn_int,fake_c_o_int):
    batch_size = fake_c_mg_int.shape[0]
    mg_label_fake = [] ; mn_label_fake = [] ; o_label_fake = []
    for i in range(batch_size):
        n_mg = fake_c_mg_int[i]+1
        n_mn = fake_c_mn_int[i]+1
        n_o = fake_c_o_int[i]+1
        mg_label_fake_i = np.array([1]*(n_mg) + [0]*(8-n_mg))
        mn_label_fake_i = np.array([1]*(n_mn) + [0]*(8-n_mn))
        o_label_fake_i = np.array([1]*(n_o) + [0]*(12-n_o))
        np.random.shuffle(mg_label_fake_i); np.random.shuffle(mn_label_fake_i); np.random.shuffle(o_label_fake_i)
        mg_label_fake.append(mg_label_fake_i.reshape(1,8,1)) ; mn_label_fake.append(mn_label_fake_i.reshape(1,8,1)) ; o_label_fake.append(o_label_fake_i.reshape(1,12,1))
    return np.vstack(mg_label_fake),np.vstack(mn_label_fake),np.vstack(o_label_fake)

def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 1, 30 , 3)
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    feature, disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty
	

def adjust_learning_rate(optimizer, epoch,initial_lr):
	lr = initial_lr * (0.95 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=501, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='adam: learning rate')
    parser.add_argument('--q_lr', type=float, default=0.000025)
    parser.add_argument('--g_lr', type=float, default=0.00005)
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
    parser.add_argument('--model_save_dir', type = str, default = './model_cwgan_mgmno/')
    parser.add_argument('--load_model', type = bool, default = False)
    parser.add_argument('--load_generator', type = str)
    parser.add_argument('--load_discriminator', type = str)
    parser.add_argument('--load_q', type = str)
    parser.add_argument('--constraint_epoch', type = int, default = 10000)
    parser.add_argument('--gen_dir', type=str, default='./gen_image_cwgan_mgmno/')
    parser.add_argument('--trainingdata', type=str, default='mgmno_100.pickle')
    parser.add_argument('--input_dim', type=str, default=512+28+1)
    opt = parser.parse_args()
    print(opt)

    job_name = '_'.join(opt.model_save_dir.split('_')[1:])[:-1]
    print(job_name)

    if not os.path.isdir(opt.gen_dir):
        os.makedirs(opt.gen_dir)
    if not os.path.isdir(opt.model_save_dir):
        os.makedirs(opt.model_save_dir)

	## Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

	## Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    net_Q = QHead_(opt)
    if cuda:
        generator.cuda()
        discriminator.cuda()
        net_Q.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()


	## Configure data loader
    train_data = np.load(opt.trainingdata, allow_pickle=True)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size = opt.batch_size, shuffle = True)

	## Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    optimizer_Q = torch.optim.Adam(net_Q.parameters(),
                                   lr=opt.q_lr, betas=(opt.b1, opt.b2))

	## Load model or Initialize
    if opt.load_model:
        generator.load_state_dict(torch.load(opt.load_generator))
        discriminator.load_state_dict(torch.load(opt.load_discriminator))
        net_Q.load_state_dict(torch.load(opt.load_q))
        print("load model ! ", opt.load_generator, opt.load_discriminator, opt.load_q)
    else:
        generator.apply(weights_init)
        print("generator weights are initialized")
        discriminator.apply(weights_init)
        print("discriminator weights are initialized")
        net_Q.apply(weights_init)
        print("net Q  weights are initialized")
	
    one = torch.FloatTensor([1])
    mone = one * -1    
	
    if cuda:
        one = one.cuda()
        mone = mone.cuda()

    for epoch in range(opt.n_epochs):
        r_mg = []
        r_mn =[]
        r_o = []
        r_c = []
        f_mg = []
        f_mn = []
        f_o = []
        f_c = []
        w = []
        for j, (imgs,label) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs = imgs.view(batch_size, 1, 30,3)
            real_imgs_noise = noising(real_imgs)
            mg_label = label[:,:8,:]
            mn_label = label[:,8:16,:]
            o_label = label[:,16:,:]
            n_mg = count_element(mg_label).reshape(batch_size,)
            n_mn = count_element(mn_label).reshape(batch_size,)
            n_o = count_element(o_label).reshape(batch_size,)
            natoms = n_mg + n_mn + n_o
            
            n_mg = n_mg -1
            n_mn = n_mn -1
            n_o = n_o -1	
            real_imgs = autograd.Variable(real_imgs.type(FloatTensor))
            real_imgs_noise = autograd.Variable(real_imgs_noise.type(FloatTensor))
            real_labels_mg = autograd.Variable(n_mg.type(LongTensor))
            real_labels_mn = autograd.Variable(n_mn.type(LongTensor))			
            real_labels_o = autograd.Variable(n_o.type(LongTensor))
            mg_label = autograd.Variable(mg_label.type(LongTensor))
            mn_label = autograd.Variable(mn_label.type(LongTensor))
            o_label = autograd.Variable(o_label.type(LongTensor))
            cell_label = autograd.Variable((natoms.type(FloatTensor))/(28.0)).unsqueeze(-1)
            
            valid = Variable(FloatTensor(np.random.uniform(0.8,1.0,size=(batch_size,1))), requires_grad = False)
            fake = Variable(FloatTensor(np.random.uniform(0,0.2,size=(batch_size,1))), requires_grad = False)
            
            
            for p in discriminator.parameters():
                p.requires_grad = True
                
                
            discriminator.zero_grad()
            net_Q.zero_grad()
            optimizer_D.zero_grad()
            optimizer_Q.zero_grad()
            
            if cuda:
                real_imgs = real_imgs.cuda()
                real_imgs_noise = real_imgs_noise.cuda()
                real_labels_mg = real_labels_mg.cuda()
                real_labels_mn = real_labels_mn.cuda()
                real_labels_o = real_labels_o.cuda()
                mg_label = mg_label.cuda()
                mn_label = mn_label.cuda()
                o_label = o_label.cuda()
                cell_label = cell_label.cuda()
                
                
            real_feature,D_real = discriminator(real_imgs)
            real_mg_label,real_mn_label,real_o_label,real_mg_cat, real_mn_cat,real_o_cat, cell_pred = net_Q(real_imgs_noise)
            D_real = D_real.mean()
            
            z = autograd.Variable(FloatTensor(np.random.normal(0,1,(batch_size, opt.latent_dim))), volatile = True)
            if cuda :
                z = z.cuda()
                
                
            fake_c_mg_int = np.random.randint(0, 8, batch_size)
            fake_c_mg = to_categorical(fake_c_mg_int,num_columns = 8)
            fake_c_mn_int = np.random.randint(0, 8, batch_size)
            fake_c_mn = to_categorical(fake_c_mn_int,num_columns = 8)
            fake_c_o_int = np.random.randint(0,12,batch_size)
            fake_c_o = to_categorical(fake_c_o_int, num_columns = 12)

            mg_label_fake,mn_label_fake,o_label_fake = make_fake_label(fake_c_mg_int, fake_c_mn_int, fake_c_o_int)

            natoms_fake = fake_c_mg_int + fake_c_mn_int + fake_c_o_int + 3
            natoms_fake = Variable(FloatTensor(natoms_fake)/(28.0)).unsqueeze(-1)
            
            if cuda:
                fake_c_mg_int = torch.tensor(fake_c_mg_int).cuda()
                fake_c_mg = fake_c_mg.cuda()
                mg_label_fake = torch.tensor(mg_label_fake).type(LongTensor).cuda()
                fake_c_mn_int = torch.tensor(fake_c_mn_int).cuda()
                fake_c_mn = fake_c_mn.cuda()
                mn_label_fake = torch.tensor(mn_label_fake).type(LongTensor).cuda()
                fake_c_o_int = torch.tensor(fake_c_o_int).cuda()
                fake_c_o = fake_c_o.cuda()			
                o_label_fake = torch.tensor(o_label_fake).type(LongTensor).cuda()
                natoms_fake = natoms_fake.cuda()	
                
                
            fake = generator(z,fake_c_mg,fake_c_mn,fake_c_o,natoms_fake)
            fake = autograd.Variable(fake)
            fake_feature, D_fake = discriminator(fake)
            
            cat_loss_mg_real = categorical_loss(real_mg_label,mg_label)
            cat_loss_mn_real = categorical_loss(real_mn_label,mn_label)
            cat_loss_o_real = categorical_loss(real_o_label, o_label)
            
            cat_loss_mg_real2 = categorical_loss(real_mg_cat,real_labels_mg)
            cat_loss_mn_real2 = categorical_loss(real_mn_cat,real_labels_mn)
            cat_loss_o_real2 = categorical_loss(real_o_cat,real_labels_o)
#            cell_loss_real = continuous_loss(cell_pred,cell_label)
            
            cat_loss_real = (cat_loss_mg_real + cat_loss_mn_real + cat_loss_o_real) + 0.3*(cat_loss_mg_real2 + cat_loss_mn_real2 + cat_loss_o_real2)
            
            r_mg.append(cat_loss_mg_real2.item())
            r_mn.append(cat_loss_mn_real2.item())
            r_o.append(cat_loss_o_real2.item())
#            r_c.append(cell_loss_real.item())
            
            
            D_real_cat = D_real - cat_loss_real
            D_real_cat.backward(mone)
            
            D_fake = D_fake.mean()			
            D_fake.backward(one)			
            
            gradient_penalty = calc_gradient_penalty(discriminator, real_imgs, fake)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            w.append(Wasserstein_D.item())
            
            optimizer_D.step()
            optimizer_Q.step()
            
            
            if j % 5 == 0 :		
                for p in discriminator.parameters():
                    p.requires_grad = False
                    
                generator.zero_grad()
                net_Q.zero_grad()
                optimizer_G.zero_grad()
                optimizer_Q.zero_grad()
                
                z = autograd.Variable(FloatTensor(np.random.normal(0,1,(batch_size, opt.latent_dim))), volatile = True)
                fake = generator(z,fake_c_mg,fake_c_mn,fake_c_o,natoms_fake)
                fake_feature, G = discriminator(fake)
                fake_mg_label, fake_mn_label, fake_o_label, fake_mg_cat, fake_mn_cat, fake_o_cat, fake_cell_pred = net_Q(fake)
                
                cat_loss_mg_fake = categorical_loss(fake_mg_label , mg_label_fake)
                cat_loss_mn_fake = categorical_loss(fake_mn_label , mn_label_fake)
                cat_loss_o_fake = categorical_loss(fake_o_label , o_label_fake)


                cat_loss_mg_fake2 = categorical_loss(fake_mg_cat, fake_c_mg_int)
                cat_loss_mn_fake2 = categorical_loss(fake_mn_cat, fake_c_mn_int)
                cat_loss_o_fake2 = categorical_loss(fake_o_cat, fake_c_o_int)
#                cell_fake = continuous_loss(fake_cell_pred, natoms_fake)
                
                f_mg.append(cat_loss_mg_fake2.item())
                f_mn.append(cat_loss_mn_fake2.item())
                f_o.append(cat_loss_o_fake2.item())
#                f_c.append(cell_fake.item())
                G = G.mean()
                
                cat_loss_fake = 0.0*(cat_loss_mg_fake + cat_loss_mn_fake + cat_loss_o_fake) + 0.3*(cat_loss_mg_fake2 + cat_loss_mn_fake2 + cat_loss_o_fake2)
                cat_loss = cat_loss_fake
                
                G_cat = G - cat_loss
                G_cat.backward(mone)
                G_cost = -G
                optimizer_Q.step()
                optimizer_G.step()

            if j == 0:
                gen_images = fake
            else:
                gen_images = torch.cat((gen_images, fake), dim = 0)
                batches_done = epoch * len(dataloader) + j

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), opt.model_save_dir+'generator_'+str(epoch))
            torch.save(discriminator.state_dict(), opt.model_save_dir+'discriminator_'+str(epoch))
            torch.save(net_Q.state_dict(), opt.model_save_dir+'Q_'+str(epoch))
            
            
        log_string = "[Epoch %d/%d] [Batch %d/%d] [W loss: %f] "  % (epoch, opt.n_epochs, j, len(dataloader),
                                                            sum(w)/len(w)) 
        
        log_string += "[real Mg : %f] [real Mn : %f] [real O : %f] [fake Mg : %f] [fake Mn : %f] [fake O : %f]" %(sum(r_mg)/len(r_mg), sum(r_mn)/len(r_mn), sum(r_o)/len(r_o),sum(f_mg)/len(f_mg), sum(f_mn)/len(f_mn), sum(f_o)/len(f_o))
	



        if epoch ==0:
            with open('train_log_'+job_name,'w') as f:
                f.write(log_string+'\n')
        else:
            with open('train_log_'+job_name,'a') as f:
                f.writelines([log_string+'\n'])	

        if epoch % 5 == 0:		
            gen_name = opt.gen_dir+'gen_images_'+str(epoch)
            tt = gen_images.cpu().detach().numpy()
            np.save(gen_name, tt)

        adjust_learning_rate(optimizer_D,epoch+1,opt.d_lr)
        adjust_learning_rate(optimizer_G,epoch+1,opt.g_lr)
        adjust_learning_rate(optimizer_Q,epoch+1,opt.q_lr)

if  __name__ == '__main__':
    print("not import")
    main()
else:
    print("import")
    pass
