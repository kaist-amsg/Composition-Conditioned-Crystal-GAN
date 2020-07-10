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

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def weights_init(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d : 
		init.xavier_uniform_(m.weight)
		init.constant_(m.bias, 0.0)
		print( m, "initialized")


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
#       n_o = (label==2).sum(dim=1)
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

class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + 8 + 8 + 12 +1
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*28),nn.ReLU(True))

        self.map1 = nn.Sequential(nn.ConvTranspose2d(128,256,(1,3),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.BatchNorm2d(512,0.8),nn.ReLU(True)) #(28,3)
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) #(28,3)
        self.cellmap = nn.Sequential(nn.Linear(84,30),nn.BatchNorm1d(30),nn.ReLU(True),nn.Linear(30,6),nn.Sigmoid())


    def forward(self, noise,c1,c2,c3,c4):
        gen_input = torch.cat((noise,c4,c1,c2,c3), -1)
        gen_input = self.l1(gen_input)
        gen_input = gen_input.view(gen_input.shape[0], 128, 28, 1)
        img = self.map1(gen_input)
        img = self.map2(img)
        img = self.map3(img)
        img = self.map4(img)
        cell_ = img.view(img.shape[0],-1)
        pos = nn.Sigmoid()(img)
        cell = self.cellmap(cell_)
        cell = cell.view(cell.shape[0],1,2,3)
        image = torch.cat((cell,pos),dim =2)
        return image

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
									nn.LeakyReLU(0.2, inplace=True),nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),
									nn.LeakyReLU(0.2,inplace=True),nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
									nn.LeakyReLU(0.2,inplace=True))
			
        self.avgpool_mg = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_mn = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_o = nn.AvgPool2d(kernel_size = (12,1))

        ds_size = opt.img_size // 2**4
	
        # Output layers
        self.feature_layer = nn.Sequential(nn.Linear(1280, 1000), nn.LeakyReLU(0.2, inplace =True), nn.Linear(1000,200),nn.LeakyReLU(0.2, inplace = True))
        self.validity = nn.Sequential(nn.Linear(200,10))

    def forward(self, img):
        B = img.shape[0]
        output = self.model(img)
	
        output_c = output[:,:,:2,:]
        output_mg = output[:,:,2:10,:]
        output_mn = output[:,:,10:18,:]
        output_o = output[:,:,18:,:]
        
        output_mg = self.avgpool_mg(output_mg)
        output_mn = self.avgpool_mn(output_mn)
        output_o = self.avgpool_o(output_o)

        output_all = torch.cat((output_c,output_mg,output_mn,output_o),dim=-2)
        output_all = output_all.view(B, -1)
	
        feature = self.feature_layer(output_all)
        validity = self.validity(feature)
        return feature, validity
    
class DHead(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output


class QHead_(nn.Module):
    def __init__(self,opt):
        super(QHead_,self).__init__()
        self.model_mg = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                nn.BatchNorm2d(512,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0)
                )

        self.model_mn = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                nn.BatchNorm2d(512,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0)
                )	

        self.model_o = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                nn.BatchNorm2d(512,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(256,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0)
                )
        self.model_cell = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (1,3), stride= 1, padding = 0),
                nn.BatchNorm2d(64,0.8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), stride = 1, padding = 0),
                nn.BatchNorm2d(64,0.8),
                nn.LeakyReLU(0.2,inplace=True))

        self.softmax = nn.Softmax2d()
        self.label_mg_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_mn_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_o_layer = nn.Sequential(nn.Linear(24,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,12),nn.Softmax())
        self.label_c_layer = nn.Sequential(nn.Linear(128,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,50),nn.BatchNorm1d(50,0.8),nn.LeakyReLU(),nn.Linear(50,1),nn.Sigmoid())
    def forward(self, image):
        cell = image[:,:,:2,:]
        mg = image[:,:,2:10,:]
        mn = image[:,:,10:18,:]
        o = image[:,:,18:,:]

        cell_output = self.model_cell(cell)
        mg_output = self.model_mg(mg)
        mn_output = self.model_mn(mn)
        o_output = self.model_o(o)
        
        cell_output_f = torch.flatten(cell_output,start_dim=1)
        mg_output_f = torch.flatten(mg_output,start_dim=1)
        mn_output_f = torch.flatten(mn_output,start_dim=1)
        o_output_f = torch.flatten(o_output,start_dim=1)

        mg_output_sm = self.softmax(mg_output)
        mn_output_sm = self.softmax(mn_output)
        o_output_sm = self.softmax(o_output)

        cell_label = self.label_c_layer(cell_output_f)
        mg_cat = self.label_mg_layer(mg_output_f)
        mn_cat = self.label_mn_layer(mn_output_f)
        o_cat = self.label_o_layer(o_output_f)
        return mg_output_sm,mn_output_sm,o_output_sm, mg_cat,mn_cat,o_cat,cell_label



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
	parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

	parser.add_argument('--unrolled_steps', type=int, default=5)
	parser.add_argument('--d_steps', type=int, default=1)
	parser.add_argument('--g_steps', type=int, default=1)
	parser.add_argument('--i_steps', type=int, default=1)
	parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
	parser.add_argument('--code_dim', type=int, default=2, help='latent code')
	parser.add_argument('--n_classes_ca', type=int, default=4, help='number of classes for dataset')
	parser.add_argument('--n_classes_mn', type=int, default=4)
	parser.add_argument('--n_classes', type=int, default=12)
	parser.add_argument('--model_save_dir', type = str, default = './model_cwgan_mgmno_200514/')
	parser.add_argument('--load_model', type = bool, default = False)
	parser.add_argument('--load_generator', type = str, default = './model_cwgan_mgmno_200513/generator_110')
	parser.add_argument('--load_discriminator', type = str, default = './model_cwgan_mgmno_200513/discriminator_110')
	parser.add_argument('--load_info', type = str, default = './model_cwgan_mgmno_200513/info_110')
	parser.add_argument('--constraint_epoch', type = int, default = 10000)
	parser.add_argument('--gen_dir', type=str, default='./gen_image_cwgan_mgmno_200514/')

	parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
	parser.add_argument('--channels', type=int, default=1, help='number of image channels')
	parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
	opt = parser.parse_args()
	print(opt)

	job_name = '_'.join(opt.model_save_dir.split('_')[1:])[:-1]
	print(job_name)

	if not os.path.isdir(opt.gen_dir):
        	os.makedirs(opt.gen_dir)
	if not os.path.isdir(opt.model_save_dir):
        	os.makedirs(opt.model_save_dir)
	# Loss functions
	adversarial_loss = torch.nn.MSELoss()
	categorical_loss = torch.nn.CrossEntropyLoss()
	continuous_loss = torch.nn.MSELoss()

	# Loss weights
	lambda_cat = 1
	lambda_con = 0.1
	lambda_constraint = 0.01

	# Initialize generator and discriminator
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

	# Initialize weights
	#generator.apply(weights_init_normal)
	#discriminator.apply(weights_init_normal)

	# Configure data loader
	train_data = np.load('./mgmno_3000.pickle', allow_pickle=True)
#	train_data = np.load('./gan_input_vo_label.pickle')
	dataloader = torch.utils.data.DataLoader(train_data, batch_size = opt.batch_size, shuffle = True)

	# Optimizers
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
	optimizer_info = torch.optim.Adam( net_Q.parameters(),
                                   lr=opt.q_lr, betas=(opt.b1, opt.b2))

	# Adversarial ground truths
	#valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
	#fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

	
	#static_code = Variable(FloatTensor(np.zeros((opt.n_classes_o**2, opt.code_dim))))

	#Load model
	if opt.load_model:
    		generator.load_state_dict(torch.load(opt.load_generator))
    		generator.eval()
    		discriminator.load_state_dict(torch.load(opt.load_discriminator))
    		discriminator.eval()
#		net_Q.load_state_dict(torch.load())
#		net_Q.eval()
    		print("load model ! ", opt.load_generator, opt.load_discriminator, opt.load_info)
	else:
			generator.apply(weights_init)
			print("generator weights are initialized")
			discriminator.apply(weights_init)
			print("discriminator weights are initialized")
			net_Q.apply(weights_init)
			print("net Q  weights are initialized")
	
#	net_Q.load_state_dict(torch.load('./model_classifier_0812/info_1000'))
#	net_Q.eval()
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
#			n_v,n_o = get_onehot(label,8,12)
#			n_v = torch.tensor(n_v)
#			n_o = torch.tensor(n_o)
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
			optimizer_info.zero_grad()
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
			fake_labels_mg_int = np.random.randint(0, 8, batch_size)
			fake_labels_mg = to_categorical(fake_labels_mg_int,num_columns = 8)
			fake_labels_mn_int = np.random.randint(0, 8, batch_size)
			fake_labels_mn = to_categorical(fake_labels_mn_int,num_columns = 8)
			fake_labels_o_int = np.random.randint(0,12,batch_size)
			fake_labels_o = to_categorical(fake_labels_o_int, num_columns = 12)
			natoms_fake = fake_labels_mg_int + fake_labels_mn_int + fake_labels_o_int + 3
			natoms_fake = Variable(FloatTensor(natoms_fake)/(28.0)).unsqueeze(-1)
			if cuda:
				fake_labels_mg_int = torch.tensor(fake_labels_mg_int).cuda()
				fake_labels_mg = fake_labels_mg.cuda()
				fake_labels_mn_int = torch.tensor(fake_labels_mn_int).cuda()
				fake_labels_mn = fake_labels_mn.cuda()
				fake_labels_o_int = torch.tensor(fake_labels_o_int).cuda()
				fake_labels_o = fake_labels_o.cuda()			
				natoms_fake = natoms_fake.cuda()	


			fake = generator(z,fake_labels_mg,fake_labels_mn,fake_labels_o,natoms_fake)
			fake = autograd.Variable(fake)
			fake_feature, D_fake = discriminator(fake)
		
			cat_loss_mg_real = categorical_loss(real_mg_label,mg_label)
			cat_loss_mn_real = categorical_loss(real_mn_label,mn_label)
			cat_loss_o_real = categorical_loss(real_o_label, o_label)
                        
			cat_loss_mg_real2 = categorical_loss(real_mg_cat,real_labels_mg)
			cat_loss_mn_real2 = categorical_loss(real_mn_cat,real_labels_mn)
			cat_loss_o_real2 = categorical_loss(real_o_cat,real_labels_o)
			cell_loss_real = continuous_loss(cell_pred,cell_label)                        

			cat_loss_real = cat_loss_mg_real + cat_loss_mn_real + cat_loss_o_real + cat_loss_mg_real2 + cat_loss_mn_real2 + cat_loss_o_real2
                        
			r_mg.append(cat_loss_mg_real2.item())
			r_mn.append(cat_loss_mn_real2.item())
			r_o.append(cat_loss_o_real2.item())
			r_c.append(cell_loss_real.item())
                        
			D_real_cat = D_real-3*cat_loss_real - cell_loss_real
			D_real_cat.backward(mone)
			
			D_fake = D_fake.mean()			
			D_fake.backward(one)			
			
			gradient_penalty = calc_gradient_penalty(discriminator, real_imgs, fake)
			gradient_penalty.backward()
			
			D_cost = D_fake - D_real + gradient_penalty
			Wasserstein_D = D_real - D_fake
			w.append(Wasserstein_D.item())
	
			optimizer_D.step()
			optimizer_info.step()
			

			if j % 5 == 0 :		
				for p in discriminator.parameters():
					p.requires_grad = False
				generator.zero_grad()
				net_Q.zero_grad()
				optimizer_G.zero_grad()
				optimizer_info.zero_grad()				

				z = autograd.Variable(FloatTensor(np.random.normal(0,1,(batch_size, opt.latent_dim))), volatile = True)	
				fake = generator(z,fake_labels_mg,fake_labels_mn,fake_labels_o,natoms_fake)
				fake_feature, G = discriminator(fake)
				fake_mg_label, fake_mn_label, fake_o_label, fake_mg_cat, fake_mn_cat, fake_o_cat, fake_cell_pred = net_Q(fake)
				
				cat_mg_fake = categorical_loss(fake_mg_cat, fake_labels_mg_int)
				cat_mn_fake = categorical_loss(fake_mn_cat, fake_labels_mn_int)
				cat_o_fake = categorical_loss(fake_o_cat, fake_labels_o_int)
				cell_fake = continuous_loss(fake_cell_pred, natoms_fake)
				
				f_mg.append(cat_mg_fake.item())
				f_mn.append(cat_mn_fake.item())
				f_o.append(cat_o_fake.item())
				f_c.append(cell_fake.item())
				G = G.mean()				

				cat_loss_fake = cat_mg_fake + cat_mn_fake + cat_o_fake
				cat_loss = cat_loss_fake
				
				G_cat = G - cat_loss - 0.3*cell_fake
				
				G_cat.backward(mone)
				G_cost = -G
				optimizer_info.step()
				optimizer_G.step()

			if j == 0:
				gen_images = fake
			else:
				gen_images = torch.cat((gen_images, fake), dim = 0)
				batches_done = epoch * len(dataloader) + j
		if epoch % 10 == 0:
			torch.save(generator.state_dict(), opt.model_save_dir+'generator_'+str(epoch))
			torch.save(discriminator.state_dict(), opt.model_save_dir+'discriminator_'+str(epoch))
			torch.save(net_Q.state_dict(), opt.model_save_dir+'info_'+str(epoch))

		log_string = "[Epoch %d/%d] [Batch %d/%d] [W loss: %f] "  % (epoch, opt.n_epochs, j, len(dataloader),
                                                            sum(w)/len(w)) 
		log_string += "[real Mg : %f] [real Mn : %f] [real O : %f] [real cell loss : %f] [fake Mg : %f] [fake Mn : %f] [fake O : %f] [fake cell loss : %f]" %(sum(r_mg)/len(r_mg), sum(r_mn)/len(r_mn), sum(r_o)/len(r_o),sum(r_c)/len(r_c),sum(f_mg)/len(f_mg), sum(f_mn)/len(f_mn), sum(f_o)/len(f_o), sum(f_c)/len(f_c))
			
		if epoch ==0:
			with open('train_log_'+job_name,'w') as f:
				f.write(log_string+'\n')
		else:
#				with open("train_log_"+job_name, 'r') as f:
#					lines= f.readlines()
#					lines.append(log_string+'\n')
			with open('train_log_'+job_name,'a') as f:
#					fsp = ''.join(lines)
#                    f.writelines([log_string+'\n')
				f.writelines([log_string+'\n'])	
#			for param_group in optimizer_D.param_groups:
#				print(param_group['lr'])
		if epoch % 5 == 0:		
			gen_name = opt.gen_dir+'gen_images_'+str(epoch)
			tt = gen_images.cpu().detach().numpy()
			np.save(gen_name, tt)

		adjust_learning_rate(optimizer_D,epoch+1,opt.d_lr)
		adjust_learning_rate(optimizer_G,epoch+1,opt.g_lr)
		adjust_learning_rate(optimizer_info,epoch+1,opt.q_lr)
if  __name__ == '__main__':
	print("not import")
	main()
else:
	print("import")
	pass
