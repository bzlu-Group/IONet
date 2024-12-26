import numpy as np
import torch
import random
import os
from Tool import read_xyzfile, mesh_data_label, read_pqr
alpha=7.0465*1e3  # ec**2*beta/s0*1e10
kbT=0.592783
pi=torch.tensor(np.pi)
cs=80    # epsilon_s
cp=2     # epsilon_p
Is=1e-5  # c=0.1 mol/l



class Data(object):
    def __init__(self, mesh_file, pqr_file, xyz_file,csv_file, device):
        self.mesh_file = mesh_file
        self.pqr_file = pqr_file
        self.csv_file = csv_file
        self.xyz_file = xyz_file
        self.centers,_, self.qs = read_pqr(self.pqr_file,device)
        self.device = device
    
    def generate_epsilon(self,train_sample_num):
        epsilon_m =torch.rand(train_sample_num)+1   #[1,2]
        epsilon_s = torch.rand(train_sample_num)*20+80   #[80,100]
        
        return epsilon_m.to(self.device), epsilon_s.to(self.device)


    def G_and_Ggrad_function(self,x,e_m):
        """
        singular change
        """ 
        s=0
        s_grad = 0
        for i,cen in enumerate(self.centers):
            ri = torch.norm(x-cen,dim=1).view(-1,1)
            s+=self.qs[i]/ri
            s_grad-=self.qs[i]*(x-cen)/(ri**3)
        G= (alpha/(4*np.pi*e_m)*s).view(-1,1)
        G_grad = (alpha/(4*np.pi*e_m)*s_grad)
        return   G,  G_grad


    def boundary_function(self, x, e_s):
        s=0
        k2=8.4869*Is/e_s
        k=torch.sqrt(k2)       
        for i,cen in enumerate(self.centers):
            ri=torch.norm(x-cen,dim=1) 
            s+=self.qs[i]*torch.exp(-k*ri)/ri
        return (alpha/(4*np.pi*e_s)*s).view(-1,1)


    def Generate_train_data(self, train_sample_num):
        """
        For each sample: training data distribution
        num_inner=100, num_out=100, num_gamma = 100, num_boundary = 100
        """
        epsilon_m, epsilon_s = self.generate_epsilon(train_sample_num)
        gamma_point, fdirection = read_xyzfile(self.xyz_file,self.device)
        inner_point, _,_, out_point, _, boundary_point, _ = mesh_data_label(self.mesh_file,self.csv_file,self.device)

        num_inner_point = len(inner_point)
        num_out_point =len(out_point)
        num_boundary_point = len(boundary_point)
        num_gamma_point = len(gamma_point)

        num_inner = int(num_inner_point/10)   
        num_out = int(num_out_point/10)
        num_gamma = int(num_boundary_point/10) 
        num_boundary = int(num_gamma_point/10)
        print('All number of training data points:')
        print('num_inner_point: {}, num_out_point: {} , num_boundary_point: {}, num_gamma_point: {}.'.format(num_inner_point,num_out_point,num_boundary_point,num_gamma_point))

        def generate(e_m, e_s):
            """
            p11,p12: inner
            p21,p22: out
            """
  
            index_inner =  random.sample(range(0,num_inner_point), num_inner)
            x_inner = inner_point[index_inner,:]

            index_out =  random.sample(range(0,num_out_point), num_out)
            x_out = out_point[index_out,:]       

            index_gamma =  random.sample(range(0,num_gamma_point), num_gamma)
            x_gamma = gamma_point[index_gamma,:]
            x_gamma_fd =  fdirection[index_gamma,:]
            G_gamma, G_gamma_grad = self.G_and_Ggrad_function(x_gamma,e_m)

            index_boundary =  random.sample(range(0,num_boundary_point), num_boundary)
            x_boundary = boundary_point[index_boundary,:]
            xb_label = self.boundary_function(x_boundary, e_s)

            sample1 = torch.hstack((torch.tile(e_m, (num_inner, 1)), torch.tile(e_s, (num_inner, 1)), x_inner))
            sample2 = torch.hstack((torch.tile(e_m, (num_out, 1)), torch.tile(e_s, (num_out, 1)), x_out))
            sample3 = torch.hstack((torch.tile(e_m, (num_gamma, 1)), torch.tile(e_s, (num_gamma, 1)), x_gamma, x_gamma_fd, G_gamma, G_gamma_grad))

            sample4 = torch.hstack((torch.tile(e_m, (num_boundary, 1)), torch.tile(e_s, (num_boundary, 1)), x_boundary, xb_label))
          
            return sample1,sample2,sample3,sample4


        for i in range(train_sample_num):
            if i==0:
                xtrain_inner, xtrain_out, xtrain_gamma, xtrain_boundary = generate(epsilon_m[i], epsilon_s[i])
            else:
                s1,s2,s3,s4 = generate(epsilon_m[i], epsilon_s[i])
                xtrain_inner = torch.vstack((xtrain_inner,s1))
                xtrain_out = torch.vstack((xtrain_out,s2))
                xtrain_gamma = torch.vstack((xtrain_gamma,s3))
                xtrain_boundary = torch.vstack((xtrain_boundary,s4))
            
        xtrain_inner = (xtrain_inner[...,0:1],xtrain_inner[...,1:2],xtrain_inner[...,2:])
        xtrain_out = (xtrain_out[...,0:1],xtrain_out[...,1:2],xtrain_out[...,2:])
        xtrain_gamma = (xtrain_gamma[...,0:1] ,xtrain_gamma[...,1:2],xtrain_gamma[...,2:5], xtrain_gamma[...,5:8],xtrain_gamma[...,8:9],xtrain_gamma[...,9:])
        xtrain_boundary = (xtrain_boundary[...,0:1],xtrain_boundary[...,1:2],xtrain_boundary[...,2:5],xtrain_boundary[...,-1:])

        return xtrain_inner, xtrain_out, xtrain_gamma, xtrain_boundary


    def enegy_data(self):
        test_epsilon_m, test_epsilon_s = torch.tensor([[2.]]).to(self.device),  torch.tensor([[80.]]).to(self.device)
        num = self.centers.shape[0]
        x = torch.hstack((torch.tile(test_epsilon_m[0], (num, 1)), torch.tile(test_epsilon_s[0], (num, 1)), self.centers))
        return (x[...,0:1],x[...,1:2],x[...,2:])

