import torch
import numpy as np
pi=3.141592653
torch.set_default_dtype(torch.float32)

class Data(object):
    def __init__(self, step, num_sample_pairs, device):
        self.device = device
        self.step = step # control the number of sensors
        self.num_sample_pairs = num_sample_pairs # 320
        self.train_domain_num = 500              
        self.train_interface_num = 500           
        self.train_boundary_num = 500           
        self.box = [-1, 1, -1, 1]        
        self.generate_sensor()

    def Generate_test_data(self):
        test_up, test_label_up, test_down, test_label_down = self.__Generate_test_data()  
        print('The number of testing data:')
        print('     X_test_up/down       : {}/{}'.format(test_label_up.shape[0],test_label_down.shape[0]))
        print('     Sensor up/down       : {}/{}/{}'.format(test_up[0].shape, test_up[1].shape, test_up[2].shape))
        print('------------------------------------------------------------------------------')
        
        return test_up, test_label_up, test_down,test_label_down

    def Generate_train_data(self):
        train1_in, train1_out, train2_in, train2_out= self.__process_train()  
        X_train_in, X_train_out, X_train_interface, X_train_b = self.__Generate_train_data(train1_in, train1_out, train2_in, train2_out) 
        
        print('The number of training data:')
        print('     X_train_inner/down   : {}/{}'.format(X_train_in[0].shape[0], X_train_out[0].shape[0]))
        print('     X_train_interface    : {}'.format(X_train_interface[0].shape[0]))
        print('     X_boundary_up/down   : {}'.format(X_train_b[0].shape[0]))    
        print('------------------------------------------------------------------------------')

        return X_train_in, X_train_out, X_train_interface, X_train_b


#######################################################################################################################################
    def __RHS(self,p1,p2,points):
        """
        p1: inner
        p2: out
        """
        tmp=(1+10*(points[:,0]**2+points[:,1]**2)).to(self.device)
        rhs  = p1/tmp**2 - p2*(points[:,0]**2+points[:,1]**2)/tmp**3
        return rhs


    def __true_solution(self,points,label):
        """
        p1: inner
        p2: out
        """
        tmp=(1+10*(points[:,0]**2+points[:,1]**2))

        if label == 'inner':
            solution  = 1/tmp

        elif label == 'out':
            solution = 2/tmp

        return solution.view(-1,1)


    def __Generate_train_data(self, train1_in, train1_out, train2_in, train2_out): 
       
        def generate(p11,p12, p21,p22):
            """
            p11,p12: inner
            p21,p22: out
            """
    
            xb, xb_label = self.SampleFromBoundary()
            x_i, f_i = self.SampleFrominterface()

            x_in, x_out = self.SampleFromDomain()
            rhs_in = self.__RHS(p11, p12, x_in).view(-1,1)
            rhs_out = self.__RHS(p21, p22, x_out).view(-1,1)

            # sensors_num
            u_sensors_in = self.__RHS( p11,p12, self.sensor_inner)        
            u_sensors_out =  self.__RHS(p21, p22, self.sensor_out)    
    
            sample1 = torch.hstack((torch.tile(u_sensors_in, (self.train_domain_num, 1)), torch.tile(u_sensors_out, (self.train_domain_num, 1)), x_in, rhs_in))
            sample2 = torch.hstack((torch.tile(u_sensors_in, (self.train_domain_num, 1)), torch.tile(u_sensors_out, (self.train_domain_num, 1)), x_out, rhs_out))
            sample3 = torch.hstack((torch.tile(u_sensors_in, (self.train_interface_num, 1)), torch.tile(u_sensors_out, (self.train_interface_num, 1)), x_i,f_i))
            sample4 = torch.hstack((torch.tile(u_sensors_in, (xb.shape[0], 1)), torch.tile(u_sensors_out, (xb.shape[0], 1)), xb, xb_label))
          
            return sample1.to(self.device),sample2.to(self.device),sample3.to(self.device),sample4.to(self.device)
           

        for i in range(train1_in.shape[0]): 
            print(i)
            if i==0:
                sample1,sample2,sample3,sample4=generate(train1_in[0], train1_out[0], train2_in[0], train2_out[0])
            else:
                s1,s2,s3,s4=generate(train1_in[i], train1_out[i], train2_in[i], train2_out[i])
                sample1= torch.vstack((sample1,s1))
                sample2= torch.vstack((sample2,s2))
                sample3= torch.vstack((sample3,s3))
                sample4= torch.vstack((sample4,s4))
 
        sample_domain1= (sample1[..., :self.num_sensor_in], sample1[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample1[..., -3:-1], sample1[...,-1].view(-1,1)) 
        sample_domain2= (sample2[..., :self.num_sensor_in], sample2[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample2[..., -3:-1], sample2[...,-1].view(-1,1))
        sample_domain3= (sample3[..., :self.num_sensor_in], sample3[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample3[..., -4:-2], sample3[..., -2:])
        sample_domain4= (sample4[..., :self.num_sensor_in], sample4[..., self.num_sensor_in:self.num_sensor_in+self.num_sensor_out], sample4[..., -3:-1], sample4[..., -1].view(-1,1))

        return sample_domain1, sample_domain2, sample_domain3, sample_domain4



    def __process_train(self): 
        """
        sample from gps for training process
        """
        sample1_in =  torch.tensor(np.random.rand(self.num_sample_pairs)*50+50).float().view(-1,1)            
        sample1_out =  torch.tensor(np.random.rand(self.num_sample_pairs)*100+1550).float().view(-1,1)    
        sample2_in =  torch.tensor(np.random.rand(self.num_sample_pairs)*50+50).float().view(-1,1)           
        sample2_out =  torch.tensor(np.random.rand(self.num_sample_pairs)*100+1550).float().view(-1,1)          

        return sample1_in.to(self.device), sample1_out.to(self.device), sample2_in.to(self.device), sample2_out.to(self.device)


    def __Generate_test_data(self): 
        p1 = 80
        p2 = 1600
        u_sensors_in = self.__RHS( p1,p2,self.sensor_inner).to(self.device)
        u_sensors_out= self.__RHS( p1,p2,self.sensor_out).to(self.device)  
       
        step = 0.02
        x = np.arange(-1, 1+step, step)
        y = np.arange(-1, 1+step, step)
        xx,yy=np.meshgrid(x,y)
        input_x=torch.tensor(xx).view(-1,1)
        input_y=torch.tensor(yy).view(-1,1)
        input=(torch.cat((input_x,input_y),1)).float()
        x=input[:,0]
        y=input[:,1]

        rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
        rr=torch.tensor(rr) 
        r=0.65**(2/3)
        location=torch.where(rr<r)[0]
        test_inner=(input[location,:]).to(self.device)
        test_inner_label = self.__true_solution(test_inner,'inner').to(self.device)
        
        location=torch.where(rr>r)[0]
        test_out=(input[location,:]).to(self.device)        
        test_out_label = self.__true_solution(test_out,'out').to(self.device)

        rhs_in = self.__RHS(p1,p2,test_inner).view(-1,1).to(self.device)
        rhs_out = self.__RHS(p1,p2,test_out).view(-1,1).to(self.device)

        x_inner = (torch.tile(u_sensors_in, (test_inner.shape[0],1)),torch.tile(u_sensors_out, (test_inner.shape[0],1)), test_inner, rhs_in)
        x_out = (torch.tile(u_sensors_in, (test_out.shape[0],1)),torch.tile(u_sensors_out, (test_out.shape[0],1)), test_out, rhs_out)

        return x_inner, test_inner_label, x_out, test_out_label


    ############################### regular components ##########################
    def generate_sensor(self):
        step = self.step
        x= np.linspace(-1,1, step)
        xx,yy=np.meshgrid(x,x)
        input_x=torch.tensor(xx).view(-1,1)
        input_y=torch.tensor(yy).view(-1,1)
        input=(torch.cat((input_x,input_y),1)).float()
        x=input[:,0]
        y=input[:,1]

        rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
        rr=torch.tensor(rr) 
        r=0.65**(2/3)
        location=torch.where(rr<r)[0]
        self.sensor_inner=((input[location,:])).float().to(self.device)
        self.num_sensor_in = self.sensor_inner.shape[0]
  
        location=torch.where(rr>r)[0]
        self.sensor_out=(input[location,:]).float().to(self.device)
        self.num_sensor_out = self.sensor_out.shape[0]
        print(self.num_sensor_in , self.num_sensor_out,'------')
        


    def SampleFrominterface(self):
        num = self.train_interface_num
        theta=2*pi*torch.rand(num,device=self.device).view(-1,1)            
        x=0.65*torch.cos(theta)**3
        y=0.65*torch.sin(theta)**3
        X=torch.cat((x,y),dim=1) 

        f1=0.65*3*torch.cos(theta)**2*(-torch.sin(theta))
        f2=0.65*3*torch.sin(theta)**2*torch.cos(theta)

        f_direction=torch.cat((f2,-f1),dim=1)
        f_direction=f_direction/torch.norm(f_direction,dim=1).view(-1,1) 
        return X.to(self.device),f_direction.to(self.device)



    def __sampleFromDomain(self):
        xmin,xmax,ymin,ymax=self.box
        x = torch.rand(15*self.train_domain_num).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(15*self.train_domain_num).view(-1,1) * (ymax - ymin) + ymin
        X =torch.cat((x,y),dim=1)
        
        return X.to(self.device)


    def SampleFromDomain(self):
        """
        training: inner and out
        """
        X=self.__sampleFromDomain()  
        x=X[:,0]
        y=X[:,1]

        rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
        rr=torch.tensor(rr).to(self.device)  

        r=0.65**(2/3)
        
        location=torch.where(rr<r)[0]
        X_in=(X[location,:])
        X_in = X_in[:self.train_domain_num,:]
        
        location=torch.where(rr>r)[0]
        X_out=(X[location,:])
        X_out = X_out[:self.train_domain_num,:]

        return  X_in.to(self.device), X_out.to(self.device)


    def SampleFromBoundary(self):
         
        xmin, xmax, ymin, ymax=self.box
        n=int(self.train_boundary_num/4)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a).to(self.device)*ymin
        P = torch.cat((a,b),dim=1)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a)*ymax
        P = torch.cat((P,torch.cat((a,b),dim=1)),dim=0)
        
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a)*xmin   
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)
        
        n = self.train_boundary_num-3*n
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a)*xmax
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)        
        
        label = self.__true_solution(P,'out')                        
        return P.to(self.device), label.to(self.device)

