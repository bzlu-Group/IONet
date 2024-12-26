import torch
import random
import numpy as np
pi=3.141592653
torch.set_default_dtype(torch.float32)

class Data(object):
    def __init__(self,  gap):
 
     
        self.sample_bounary_points = torch.tensor(np.loadtxt(open('GRF-generate/train_data_257/sample_boundary_points.csv','rb'),delimiter=',')).float() # num*dim
        self.train_domain_num = 1000              
        self.train_interface_num = 100           
        self.train_boundary_num = 500            
        self.box = [0, 1, 0, 1]        
        self.gap = gap                           

 
    def Generate_test_data(self):
        test_up,  test_down = self.__Generate_test_data()  
        print('The number of testing data:')
        print('     X_test_up/down       : {}/{}'.format(test_up[2].shape[0],test_up[2].shape[0]))
        print('     Sensor up/down       : {}/{}'.format(test_up[0].shape[1], test_up[1].shape[1]))
      
        print('------------------------------------------------------------------------------')

        return test_up, test_down

    def Generate_train_data(self, sample_num):
        train_u, train_d, train_a = self.__gaussian_process_train(sample_num)  
        X_train_up, X_train_down, X_train_interface, X_train_b_up, X_train_b_down = self.__Generate_train_data(train_u, train_d, train_a) 
        
        print('The number of training data:')
        print('     X_train_up/down      : {}/{}'.format(X_train_up[0].shape[0], X_train_down[0].shape[0]))
        print('     X_train_interface    : {}'.format(X_train_interface[0].shape[0]))
        print('     X_boundary_up/down   : {}/{}'.format(X_train_b_up[0].shape[0], X_train_b_down[0].shape[0]))    
        print('------------------------------------------------------------------------------')

        return X_train_up, X_train_down, X_train_interface, X_train_b_up, X_train_b_down 

 
    def __Generate_train_data(self, gps_u, gps_d, gps_a): 
       
        def generate(gp_u, gp_d, a1, a2):
        
            xb_up, labelb_up, xb_down, labelb_down = self.SampleFromBoundary(gp_u, gp_d)
            x_i = self.SampleFrominterface()
            x_up, x_down = self.SampleFromDomain()
      
            u_sensors_up = gp_u[0:-1: self.gap]             
            u_sensors_down = gp_d[0:-1: self.gap]
            self.sensors_nup=u_sensors_up.size()[0]
            self.sensors_ndown=u_sensors_down.size()[0]

            sample1 = torch.hstack((torch.tile(u_sensors_up, (self.train_domain_num, 1)), 
                                    torch.tile(u_sensors_down, (self.train_domain_num, 1)), 
                                    torch.tile(a1, (self.train_domain_num, 1)), 
                                    torch.tile(a2, (self.train_domain_num, 1)), 
                                    x_up))
            sample2 = torch.hstack((torch.tile(u_sensors_up, (self.train_domain_num, 1)), 
                                    torch.tile(u_sensors_down, (self.train_domain_num, 1)), 
                                    torch.tile(a1, (self.train_domain_num, 1)),
                                    torch.tile(a2, (self.train_domain_num, 1)),
                                    x_down))
            sample3 = torch.hstack((torch.tile(u_sensors_up, (self.train_interface_num, 1)), 
                                    torch.tile(u_sensors_down, (self.train_interface_num, 1)), 
                                    torch.tile(a1, (self.train_interface_num, 1)), 
                                    torch.tile(a2, (self.train_interface_num, 1)), 
                                    x_i))
     
            sample4 = torch.hstack((torch.tile(u_sensors_up, (xb_up.shape[0], 1)), 
                                    torch.tile(u_sensors_down, (xb_up.shape[0], 1)), 
                                    torch.tile(a1, (xb_up.shape[0], 1)),
                                    torch.tile(a2, (xb_up.shape[0], 1)),
                                    xb_up, 
                                    labelb_up))
            sample5 = torch.hstack((torch.tile(u_sensors_up, (xb_down.shape[0], 1)), 
                                    torch.tile(u_sensors_down, (xb_down.shape[0], 1)), 
                                    torch.tile(a1, (xb_down.shape[0], 1)), 
                                    torch.tile(a2, (xb_down.shape[0], 1)), 
                                    xb_down, 
                                    labelb_down))
            
            return sample1 ,sample2,sample3,sample4,sample5
           
        sample1,sample2,sample3,sample4,sample5 = [],[],[],[],[]
        for i in range(gps_u.shape[0]):  
            if i%200==0:
                print('train data sample num:', i+1)
            s1,s2,s3,s4,s5=generate(gps_u[i],gps_d[i], gps_a[0][i], gps_a[1][i])
            sample1.append(s1)
            sample2.append(s2)
            sample3.append(s3)
            sample4.append(s4)
            sample5.append(s5)

        sample1, sample2, sample3, sample4, sample5 = torch.vstack(sample1), torch.vstack(sample2), torch.vstack(sample3), torch.vstack(sample4), torch.vstack(sample5)

        sample_domain1= (sample1[..., :self.sensors_nup], 
                         sample1[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                         sample1[..., -4:-3],
                         sample1[..., -3:-2],
                         sample1[..., -2:]) 
        sample_domain2= (sample2[..., :self.sensors_nup], 
                         sample2[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                         sample2[..., -4:-3],
                         sample2[..., -3:-2],
                         sample2[..., -2:]) 
        sample_domain3= (sample3[..., :self.sensors_nup], 
                         sample3[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                         sample3[..., -4:-3],
                         sample3[..., -3:-2],
                         sample3[..., -2:])
        sample_domain4= (sample4[..., :self.sensors_nup], 
                         sample4[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                         sample4[..., -5:-4],
                         sample4[..., -4:-3],
                         sample4[..., -3:-1],
                         sample4[..., -1].view(-1,1))
        sample_domain5= (sample5[..., :self.sensors_nup], 
                         sample5[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                         sample5[..., -5:-4],
                         sample5[..., -4:-3],
                         sample5[..., -3:-1],
                         sample5[..., -1:].view(-1,1))

  
        return sample_domain1,sample_domain2,sample_domain3,sample_domain4,sample_domain5

    def __gaussian_process_train(self, sample_num): 
        """
        sample from gps for training process!
        """
 
        sample_data = torch.tensor(np.loadtxt(open('GRF-generate/train_data_257/train_data_samples_{}.csv'.format(sample_num),'rb'),delimiter=',')).float()  # num*dim
        
 
        sample_coef_a = torch.tensor(np.loadtxt('train_coef_a_{}.txt'.format(sample_num))).float()  # 2*num

        index_up = torch.where(self.sample_bounary_points[:,1]>0.5)[0]
        index_down = torch.where(self.sample_bounary_points[:,1]<=0.5)[0]
        sample_up = sample_data[:,index_up]       # shape: N*features 
        sample_down =sample_data[:,index_down]    # shape: N*features

        return sample_up, sample_down, sample_coef_a


    def __Generate_test_data(self): 
        """
        sample from gps for test process!
        """ 
 
        sample_boundary_features = torch.tensor(np.loadtxt(open('GRF-generate/test_data_257/test_data_samples_100.csv','rb'),delimiter=',')).float()
        ind_up = torch.where(self.sample_bounary_points[:,1]>0.5)[0]
        ind_down = torch.where(self.sample_bounary_points[:,1]<=0.5)[0]
        u_sensors_up=(sample_boundary_features[:,ind_up])[:,0:-1:self.gap]   
        u_sensors_down=(sample_boundary_features[:,ind_down])[:,0:-1:self.gap]  
       
 
        test_points = torch.tensor(np.loadtxt('GRF-generate/test_data_257/test_points_100.txt', dtype=np.float32)).float()
 
        test_labels =  torch.tensor(np.loadtxt('GRF-generate/test_data_257/test_Nlabels_100.txt', dtype=np.float32)).float() # (100, 289)
  
        test_coef_a =  torch.tensor(np.loadtxt('GRF-generate/test_data_257/test_coef_a.txt', dtype=np.float32)).float() # (2,100)

        index_up =  torch.where(test_points[:,1]>0.5)[0]
        index_down = torch.where(test_points[:,1]<=0.5)[0]
        test_points_up = test_points[index_up,:]
        test_points_down = test_points[index_down,:]

        up =[]
        down = []

        for i in range(sample_boundary_features.shape[0]):
            up.append(torch.hstack((torch.tile(u_sensors_up[i], (test_points_up.shape[0], 1)), 
                                    torch.tile(u_sensors_down[i], (test_points_up.shape[0], 1)), 
                                    torch.tile(test_coef_a[0][i], (test_points_up.shape[0], 1)), 
                                    torch.tile(test_coef_a[1][i], (test_points_up.shape[0], 1)), 
                                    test_points_up, 
                                    test_labels[i][index_up].view(-1,1))))
            down.append(torch.hstack((torch.tile(u_sensors_up[i], (test_points_down.shape[0], 1)), 
                                      torch.tile(u_sensors_down[i], (test_points_down.shape[0], 1)), 
                                      torch.tile(test_coef_a[0][i], (test_points_down.shape[0], 1)), 
                                      torch.tile(test_coef_a[1][i], (test_points_down.shape[0], 1)), 
                                      test_points_down, 
                                      test_labels[i][index_down].view(-1,1))))               


        up, down = torch.vstack(up), torch.vstack(down)
       
        x_up = (up[..., :self.sensors_nup],
                up[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                up[..., -5:-4], 
                up[..., -4:-3], 
                up[..., -3:-1], 
                up[..., -1:])
        x_down = (down[..., :self.sensors_nup], 
                  down[..., self.sensors_nup:self.sensors_nup+self.sensors_ndown], 
                  down[..., -5:-4], 
                  down[..., -4:-3], 
                  down[..., -3:-1], 
                  down[..., -1:])

        return x_up, x_down


 
    def SampleFrominterface(self):
        """
        L : the center of sphere
        output: boundary point and related f_direction 
        """
 
        x = torch.rand(self.train_interface_num).view(-1,1)
        y = torch.ones_like(x)*0.5
        X= torch.cat((x,y), dim=1)  

        return X 


    def __sampledomain(self):
        xmin,xmax,ymin,ymax = self.box
        x = torch.rand(3*self.train_domain_num).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(3*self.train_domain_num).view(-1,1) * (ymax - ymin) + ymin
        X =torch.cat((x,y),dim=1)
        
        return X


    def SampleFromDomain(self):
        """
        training: up and down
        """
        X=self.__sampledomain()      
        location=torch.where(X[:,1]>0.5)[0]
        X_up=X[location,:]
        X_up=X_up[:self.train_domain_num,:]
  
        location=torch.where(X[:,1]<0.5)[0]
        X_down=X[location,:]      
        X_down = X_down[:self.train_domain_num,:]

        return X_up,  X_down


    def SampleFromBoundary(self, gp_u, gp_d):
        """
        input: gp_u and gp_d

        """ 
        index_up = torch.where(self.sample_bounary_points[:,1]>0.5)[0]
        index_down = torch.where(self.sample_bounary_points[:,1]<=0.5)[0]

        index = random.sample(range(0,len(index_up)), self.train_boundary_num)
        x_up = (self.sample_bounary_points[index_up,:])[index,:]
        label_up = gp_u[index]

        index = random.sample(range(0,len(index_down)), self.train_boundary_num)
        x_down = (self.sample_bounary_points[index_down,:])[index,:]
        label_down = gp_d[index]

        return x_up, label_up.view(-1,1), x_down, label_down.view(-1,1)


