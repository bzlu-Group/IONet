 
import torch
import torch.nn as nn
import numpy as np
import random
from Tools import To_tensor, Get_batch
from Net_type import IONet
import torch.optim as optim
import time,os
from Tools import grad
 
lr= 1e-3
epochs = 40000
print_epoch = 500
data_file = 'data/'  
width = 100    
p = 30
batch_size = 2500
 

N = 998
xmin,xmax = 0,1
h = (xmax-xmin)/(N+1)
mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  

gamma = 0.5 

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

import os 

 
 
def Generate_test_data(data_file, gamma):
    """
    input: 
    output: test data (sensor left, sensor right, test x, test label)
    """
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    test_coef_sensor_value = np.loadtxt(data_file+'test_coef_sensor_value.txt')

    sensor_l_index = np.where(sensor_location <= gamma)[0]
    sensor_r_index = np.where(sensor_location > gamma)[0]   
    sensor_num1 = len(sensor_l_index)
    sensor_num2 = len(sensor_r_index)    

    sensor_num = sensor_num1+sensor_num2

    sensors_l = test_coef_sensor_value[:,sensor_l_index]
    sensors_r = test_coef_sensor_value[:,sensor_r_index]

    test_mesh_location = np.loadtxt(data_file+'test_mesh.txt')
    test_u_pre_solution = np.loadtxt(data_file+'test_u_solution.txt')
    x_index_l = np.where(test_mesh_location <= gamma)[0]
    x_index_r = np.where(test_mesh_location > gamma)[0]   

    x_l = test_mesh_location[x_index_l]
    x_r = test_mesh_location[x_index_r]
    label_l = test_u_pre_solution[:,x_index_l]
    label_r = test_u_pre_solution[:,x_index_r]

    x_left = []
    x_right =[]
    for i in range(test_coef_sensor_value.shape[0]):
        x_left.append(np.hstack([np.tile(sensors_l[i],  (x_l.shape[0], 1)), 
                        np.tile(sensors_r[i], (x_l.shape[0], 1)),
                        x_l[:,None], 
                        label_l[i][:,None]]))

        x_right.append(np.hstack([np.tile(sensors_l[i], (x_r.shape[0], 1)),
                            np.tile(sensors_r[i], (x_r.shape[0], 1)),
                            x_r[:,None],
                            label_r[i][:,None]]))

    x_left =np.vstack(x_left)
    x_right = np.vstack(x_right)
    print('test data shape',x_left.shape, x_right.shape)

    x_left = (x_left[:, :sensor_num1], 
                   x_left[:,  sensor_num1: sensor_num],
                   x_left[:,-2][:,None], 
                   x_left[:,-1][:,None])

    x_right = (x_right[:, :sensor_num1], 
                   x_right[:,  sensor_num1: sensor_num],
                   x_right[:,-2][:,None], 
                   x_right[:,-1][:,None])


    x_left = To_tensor(x_left)
    x_right = To_tensor(x_right)

    return x_left, x_right, sensor_num1, sensor_num2

X_test_l,  X_test_r, sensor_num1, sensor_num2 =  Generate_test_data(data_file, gamma)
# datatype change 
X_test_l = Get_batch(X_test_l,[],50000, device=device)
X_test_r = Get_batch(X_test_r,[],50000, device=device)


 
def Generate_train_data(data_file, gamma,  p):
 
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    sensor_l_index = np.where(sensor_location <= gamma)[0]
    sensor_r_index = np.where(sensor_location > gamma)[0]   
    sensor_num1 = len(sensor_l_index)
    sensor_num2 = len(sensor_r_index)   

    sensor_num =sensor_num1+sensor_num2

    train_coef_sensor_value=np.loadtxt(data_file+'train_coef_sensor_value.txt')
    train_coef_sensor_l = train_coef_sensor_value[:,sensor_l_index]
    train_coef_sensor_r = train_coef_sensor_value[:,sensor_r_index]


    train_mesh_location = np.loadtxt(data_file+'train_mesh.txt')
    x_index_l = np.where(train_mesh_location <= gamma)[0]
    x_index_r = np.where(train_mesh_location > gamma)[0]   

    x_l = train_mesh_location[x_index_l]
    x_r = train_mesh_location[x_index_r]
    train_u_pre_solution = np.loadtxt(data_file+'train_u_solution.txt')
    label_l = train_u_pre_solution[:,x_index_l]
    label_r = train_u_pre_solution[:,x_index_r]

 
    pp=int(p/2)

    def generate(s_l, s_r, label_l, label_r):  
        index = random.sample(range(len(x_l)), pp)
        sample_left=np.hstack([np.tile(s_l, ( pp, 1)),
                                np.tile(s_r, ( pp, 1)),
                                x_l[index].reshape(-1,1),
                                label_l[index].reshape(-1,1)])

        index = random.sample(range(len(x_r)), pp)
        sample_right=np.hstack([np.tile(s_l, ( pp, 1)),
                                np.tile(s_r, ( pp, 1)),
                                x_r[index].reshape(-1,1),
                                label_r[index].reshape(-1,1)])
                              
        return  (sample_left, sample_right)

    sample_left,  sample_right = [], []


    for i in range(0,train_coef_sensor_value.shape[0]):      
        if i%200==0:
            print(i)
        sample_domain = generate(train_coef_sensor_l[i],  train_coef_sensor_r[i], label_l[i],label_r[i])
                                                      
        sample_left.append(sample_domain[0])
        sample_right.append(sample_domain[1])
        
 
    sample_left, sample_right = np.vstack(sample_left), np.vstack(sample_right)
   

    sample_left      =  ( sample_left[...,:sensor_num1],  sample_left[...,sensor_num1:sensor_num],  
                         sample_left[...,-2:-1], sample_left[...,-1:])   
    sample_right      =  ( sample_right[...,:sensor_num1],  sample_right[...,sensor_num1:sensor_num], 
                          sample_right[...,-2:-1], sample_right[...,-1:])   


    x_train_l = To_tensor(sample_left)
    x_train_r = To_tensor(sample_right)

    
    return (x_train_l,  x_train_r)

sample_domain = Generate_train_data(data_file, gamma,  p=p)

 
model = IONet(sensor_num1, sensor_num2, m=width, actv=nn.ReLU()).to(device)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

 
optimizer=optim.Adam(model.parameters(),lr=lr)

t0=time.time()

for epoch in range(epochs):
 
    optimizer.zero_grad()
    input_left = Get_batch(sample_domain[0], requires_grad_remark=[2], batch_size=batch_size, device=device)
    loss_l = nn.MSELoss()(model(input_left,'l'), input_left[-1])
    input_right = Get_batch(sample_domain[1], requires_grad_remark=[2], batch_size=batch_size, device=device)
    loss_r = nn.MSELoss()(model(input_right,'r'), input_right[-1])
    loss = loss_l+loss_r
    loss.backward(retain_graph=True)
    optimizer.step()


    if epoch%print_epoch==0:
        print('epoch {}: training loss'.format(epoch), loss.item(),optimizer.param_groups[0]['lr'])
        print(loss_l.item(), loss_r.item())
        pre_label_l = model(X_test_l,label='l')
        pre_label_r = model(X_test_r,label='r')
        relative_l2 =((X_test_l[-1]-pre_label_l)**2).sum()+((pre_label_r-X_test_r[-1])**2).sum()
        relative_l2 = torch.sqrt(relative_l2/(((X_test_l[-1])**2).sum()+((X_test_r[-1])**2).sum()))
        print('Rela L2 loss is: ', relative_l2.item())
        print('\n')

    if (epoch+1)%int(epochs/100)==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

print('time',time.time()-t0)
if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
torch.save(model,'model/'+'dd_ion'+'_'+str(width)+'_'+str(epoch)+'.pkl')
