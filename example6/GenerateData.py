import numpy as np
from pyDOE import lhs
import torch
torch.set_default_dtype(torch.float32)
pi=3.141592653


class Data(object):
    def __init__(self, num_train_sample,sensor_num):
        self.num_train_sample = num_train_sample
        self.sensor_num = sensor_num
        self.train_domain_num = 500              
        self.train_interface_num = 500           
        self.train_boundary_num = 500  
        self.test_num = 5000        
        self.generate_sensor()


 
    def Generate_test_data(self,device):
        test_inner, test_label_in, test_out, test_label_down = self.__Generate_test_data(device)  
        print('The number of testing data:')
        print('     X_test_inner/down    : {}/{}'.format(test_label_in.shape[0],test_label_down.shape[0]))
        print('     Sensor up/down       : {}/{}/{}'.format(test_inner[0].shape, test_inner[1].shape, test_inner[2].shape))
        print('------------------------------------------------------------------------------')
        
        return test_inner, test_label_in, test_out, test_label_down

    def Generate_train_data(self):
        train_in, train_out= self.__process_train()  
        X_train_in, X_train_out, X_train_interface, X_train_b= self.__Generate_train_data(train_in, train_out) 
        
        print('The number of training data:')
        print('     X_train_inner/down   : {}/{}'.format(X_train_in[0].shape[0], X_train_out[0].shape[0]))
        print('     X_train_interface    : {}'.format(X_train_interface[0].shape[0]))
        print('     X_boundary_up/down   : {}'.format(X_train_b[0].shape[0]))    
        print('------------------------------------------------------------------------------')

        return X_train_in, X_train_out, X_train_interface, X_train_b


    def __RHS(self,p1,p2,X,label):
        if label == 'inner':
            rhs  = -p1*torch.sin(X[:,0:1])*torch.sin(X[:,1:2])*torch.sin(X[:,2:3])*torch.sin(X[:,3:4])*torch.sin(X[:,4:5])*torch.sin(X[:,5:6])
        elif label == 'out':
            rhs =  -p2*torch.exp(X[:,0:1]+X[:,1:2]+X[:,2:3]+X[:,3:4]+X[:,4:5]+X[:,5:6])

        return rhs.float().view(-1,)


    def __true_solution(self, X, label):
        """
        p1: inner
        p2: out
        """
        if label == 'inner':
            solution  = torch.sin(X[:,0:1])*torch.sin(X[:,1:2])*torch.sin(X[:,2:3])*torch.sin(X[:,3:4])*torch.sin(X[:,4:5])*torch.sin(X[:,5:6])

        elif label == 'out':
            solution =  torch.exp(X[:,0:1]+X[:,1:2]+X[:,2:3]+X[:,3:4]+X[:,4:5]+X[:,5:6])

        return solution.view(-1,1)


    def __normal_u(self,X):
        x0, x1, x2, x3, x4, x5 = X[:,0:1], X[:,1:2], X[:,2:3] ,X[:,3:4],X[:,4:5],X[:,5:6]
        dist = np.sqrt(x0**2 + x1**2 + x2**2+x3**2 + x4**2 + x5**2)

        uni = np.cos(x0)*np.sin(x1)*np.sin(x2)*np.sin(x3)*np.sin(x4)*np.sin(x5)*x0\
        +np.sin(x0)*np.cos(x1)*np.sin(x2)*np.sin(x3)*np.sin(x4)*np.sin(x5)*x1\
        +np.sin(x0)*np.sin(x1)*np.cos(x2)*np.sin(x3)*np.sin(x4)*np.sin(x5)*x2\
        +np.sin(x0)*np.sin(x1)*np.sin(x2)*np.cos(x3)*np.sin(x4)*np.sin(x5)*x3\
        +np.sin(x0)*np.sin(x1)*np.sin(x2)*np.sin(x3)*np.cos(x4)*np.sin(x5)*x4\
        +np.sin(x0)*np.sin(x1)*np.sin(x2)*np.sin(x3)*np.sin(x4)*np.cos(x5)*x5
        uni = uni/dist
    
        uno = (x0+x1+x2+x3+x4+x5)*np.exp(x0+x1+x2+x3+x4+x5)
        uno = uno/dist
        return (1e-3*uno-uni).view(-1,1)


    def __Generate_train_data(self, train_in, train_out):    
        def generate(p1,p2):
            """
            p11,p12: inner
            p21,p22: out
            """
            xb, xb_label = self.SampleFromBoundary()
            x_i,ud,un = self.SampleFrominterface()

            x_in, x_out = self.SampleFromDomain()
            rhs_in = self.__RHS(p1, p2, x_in,'inner').view(-1,1)
            rhs_out = self.__RHS(p1, p2, x_out,'out').view(-1,1)

            u_sensors_in = self.__RHS(p1, p2, self.sensor_inner,'inner')        
            u_sensors_out =  self.__RHS(p1, p2, self.sensor_out,'out')    
            
            sample1 = torch.hstack((torch.tile(u_sensors_in, (self.train_domain_num, 1)), torch.tile(u_sensors_out, (self.train_domain_num, 1)), x_in, rhs_in))
            sample2 = torch.hstack((torch.tile(u_sensors_in, (self.train_domain_num, 1)), torch.tile(u_sensors_out, (self.train_domain_num, 1)), x_out, rhs_out))
            sample3 = torch.hstack((torch.tile(u_sensors_in, (self.train_interface_num, 1)), torch.tile(u_sensors_out, (self.train_interface_num, 1)), x_i, ud, un ))
            sample4 = torch.hstack((torch.tile(u_sensors_in, (xb.shape[0], 1)), torch.tile(u_sensors_out, (xb.shape[0], 1)), xb, xb_label))
          
            return sample1,sample2,sample3,sample4
           

        for i in range(train_in.shape[0]): 
            if i==0:
                sample1,sample2,sample3,sample4=generate(train_in[0], train_out[0])
            else:
                s1, s2, s3, s4 = generate(train_in[i], train_out[i])
                sample1= torch.vstack((sample1,s1))
                sample2= torch.vstack((sample2,s2))
                sample3= torch.vstack((sample3,s3))
                sample4= torch.vstack((sample4,s4))
 
        sample_domain1= (sample1[..., :self.num_sensor_in], sample1[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample1[..., -7:-1], sample1[...,-1].view(-1,1)) 
        sample_domain2= (sample2[..., :self.num_sensor_in], sample2[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample2[..., -7:-1], sample2[...,-1].view(-1,1))
        sample_domain3= (sample3[..., :self.num_sensor_in], sample3[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample3[..., -8:-2], sample3[..., -2:-1],sample3[...,-1].view(-1,1))
        sample_domain4= (sample4[..., :self.num_sensor_in], sample4[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample4[..., -7:-1], sample4[..., -1].view(-1,1))
   

        return sample_domain1,sample_domain2,sample_domain3,sample_domain4


    def __process_train(self): 
        sample_in  =  torch.tensor(np.random.rand(self.num_train_sample)*9-10).float().view(-1,1)          
        sample_out =  torch.tensor(np.random.rand(self.num_train_sample)*9+1).float().view(-1,1)         
           
        return sample_in, sample_out


    def __Generate_test_data(self,device): 
        p1 = -6
        p2 = 6
        u_sensors_in = self.__RHS( p1,p2,self.sensor_inner,'inner').to(device)
        u_sensors_out= self.__RHS( p1,p2,self.sensor_out,'out').to(device)
       
        X_domain=self.__sampleFromDomain(self.test_num)  
        index =torch.where(torch.norm(X_domain,dim=-1)<0.5)[0]
        test_inner =X_domain[index][:self.test_num,:].to(device)
        test_inner_label = self.__true_solution(test_inner,'inner').to(device)
                
        index =torch.where(torch.norm(X_domain,dim=-1)>0.5)[0]
        test_out = X_domain[index][:self.test_num,:].to(device)
        test_out_label = self.__true_solution(test_out,'out').to(device)


        x_inner = (torch.tile(u_sensors_in, (test_inner.shape[0],1)),torch.tile(u_sensors_out, (test_inner.shape[0],1)), test_inner)
        x_out = (torch.tile(u_sensors_in, (test_out.shape[0],1)),torch.tile(u_sensors_out, (test_out.shape[0],1)), test_out)

        return x_inner, test_inner_label, x_out, test_out_label


    ############################### regular components ##########################
    def generate_sensor(self):
        sensor_locations=self.__sampleFromDomain(self.sensor_num)
        index =torch.where(torch.norm(sensor_locations,dim=-1)<0.5)[0]
        self.sensor_inner =sensor_locations[index][:self.sensor_num,:]
        index =torch.where(torch.norm(sensor_locations,dim=-1)>0.5)[0]
        self.sensor_out = sensor_locations[index][:self.sensor_num,:]
        self.num_sensor_in = self.sensor_inner.shape[0]
        self.num_sensor_out = self.sensor_out.shape[0]


    def SampleFrominterface(self):
        para_ij = torch.tensor(lhs(6, self.train_interface_num)).float()
        r = 0.5
        t1 = para_ij[:,1:2]*np.pi
        t2 = para_ij[:,2:3]*np.pi
        t3 = para_ij[:,3:4]*np.pi
        t4 = para_ij[:,4:5]*np.pi
        t5 = para_ij[:,5:6]*2.0*np.pi
        X = 0.0*para_ij
        X[:, 0:1] = r*torch.cos(t1)
        X[:, 1:2] = r*torch.sin(t1)*torch.cos(t2)
        X[:, 2:3] = r*torch.sin(t1)*torch.sin(t2)*torch.cos(t3)
        X[:, 3:4] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.cos(t4)
        X[:, 4:5] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.cos(t5)
        X[:, 5:6] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.sin(t5)

        u_dirich = self.__true_solution(X,'out') -self.__true_solution(X,'inner')
        u_numan =  self.__normal_u(X)
        return X, u_dirich, u_numan



    def __sampleFromDomain(self,num):
        para_domain = torch.tensor(lhs(6, 5*num)).float()
        r = np.sqrt(para_domain[:,0:1]*0.36)
        t1 = para_domain[:,1:2]*np.pi
        t2 = para_domain[:,2:3]*np.pi
        t3 = para_domain[:,3:4]*np.pi
        t4 = para_domain[:,4:5]*np.pi
        t5 = para_domain[:,5:6]*2.0*np.pi
        X = 0.0*para_domain
        X[:, 0:1] = r*torch.cos(t1)
        X[:, 1:2] = r*torch.sin(t1)*torch.cos(t2)
        X[:, 2:3] = r*torch.sin(t1)*torch.sin(t2)*torch.cos(t3)
        X[:, 3:4] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.cos(t4)
        X[:, 4:5] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.cos(t5)
        X[:, 5:6] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.sin(t5)

        return X


    def SampleFromDomain(self):
        X_domain=self.__sampleFromDomain(self.train_domain_num)  
        index =torch.where(torch.norm(X_domain,dim=-1)<0.5)[0]
        X_inner =X_domain[index][:self.train_domain_num,:]
        index =torch.where(torch.norm(X_domain,dim=-1)>0.5)[0]
        X_out = X_domain[index][:self.train_domain_num,:]

        return  X_inner, X_out
    

    def SampleFromBoundary(self):
        
        para_bd = torch.tensor(lhs(6, self.train_boundary_num)).float()
        r = 0.6
        t1 = para_bd[:,1:2]*np.pi
        t2 = para_bd[:,2:3]*np.pi
        t3 = para_bd[:,3:4]*np.pi
        t4 = para_bd[:,4:5]*np.pi
        t5 = para_bd[:,5:6]*2.0*np.pi
        X = 0.0*para_bd
        X[:, 0:1] = r*torch.cos(t1)
        X[:, 1:2] = r*torch.sin(t1)*torch.cos(t2)
        X[:, 2:3] = r*torch.sin(t1)*torch.sin(t2)*torch.cos(t3)
        X[:, 3:4] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.cos(t4)
        X[:, 4:5] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.cos(t5)
        X[:, 5:6] = r*torch.sin(t1)*torch.sin(t2)*torch.sin(t3)*torch.sin(t4)*torch.sin(t5)
                
        label = self.__true_solution(X,'out')                        
        return X, label

