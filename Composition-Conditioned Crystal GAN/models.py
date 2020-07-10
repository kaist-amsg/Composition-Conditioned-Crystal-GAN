import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + 8 + 8 + 12 +1
        self.input_dim = opt.input_dim # 512+28+1

        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*28),nn.ReLU(True))
        self.map1 = nn.Sequential(nn.ConvTranspose2d(128,256,(1,3),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.BatchNorm2d(512,0.8),nn.ReLU(True)) #(28,3)
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) #(28,3)
        self.cellmap = nn.Sequential(nn.Linear(84,30),nn.BatchNorm1d(30),nn.ReLU(True),nn.Linear(30,6),nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise,c1,c2,c3,c4):
        gen_input = torch.cat((noise,c4,c1,c2,c3), -1)
        h = self.l1(gen_input)
        h = h.view(h.shape[0], 128, 28, 1)
        h = self.map1(h)
        h = self.map2(h)
        h = self.map3(h)
        h = self.map4(h)

        h_flatten = h.view(h.shape[0],-1)
        pos = self.sigmoid(h)
        cell = self.cellmap(h_flatten)
        cell = cell.view(cell.shape[0],1,2,3)
        return torch.cat((cell,pos),dim =2)



class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                   nn.LeakyReLU(0.2, inplace=True),nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),                                                                             nn.LeakyReLU(0.2,inplace=True),nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),                                                                               nn.LeakyReLU(0.2,inplace=True))

        self.avgpool_mg = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_mn = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_o = nn.AvgPool2d(kernel_size = (12,1))

        self.feature_layer = nn.Sequential(nn.Linear(1280, 1000), nn.LeakyReLU(0.2, inplace =True), nn.Linear(1000,200),nn.LeakyReLU(0.2, inplace = True))
        self.output = nn.Sequential(nn.Linear(200,10))

    def forward(self, x):
        B = x.shape[0]
        output = self.model(x)

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
        return feature, self.output(feature)


class QHead_(nn.Module):
    def __init__(self,opt):
        super(QHead_,self).__init__()
        self.model_mg = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                      nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),                                                                                                                                                     nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_mn = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                      nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_o = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                     nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                     nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                     nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_cell = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (1,3), stride= 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True),
                                        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), stride = 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True))

        self.softmax = nn.Softmax2d()
        self.label_mg_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                            nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_mn_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                            nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_o_layer = nn.Sequential(nn.Linear(24,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                           nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,12),nn.Softmax())
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

#if __name__ == '__main__':
#    pass



