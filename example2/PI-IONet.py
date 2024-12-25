# %%
import torch
import torch.nn as nn
import numpy as np
import random
from Tools import To_tensor
from Net_type import IONet
import torch.optim as optim
import time, os
from Tools import grad, Get_batch

# %%
lr= 1e-3
epochs = 40000
print_epoch = 200
data_file = 'data_05_l025_l0201/'   
width = 70    
p = 20
batch_size = 1000
 
N = 998
xmin, xmax = 0, 1
h = (xmax-xmin)/(N+1)
mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  

gamma = 0.5  

## device
if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'
 


def Generate_test_data(data_file, gamma, device):
    """
    input: 
    output: test data (sensor left, sensor right, test x, test label)
    """
    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    sensor_l_index = np.where(sensor_location <= gamma)[0]
    sensor_r_index = np.where(sensor_location > gamma)[0]   
    sensor_num1 = len(sensor_l_index)
    sensor_num2 = len(sensor_r_index)    


    test_rhs_sensor_value = np.loadtxt(data_file+'test_rhs_sensor_value.txt')
    test_b_sensor_value = np.loadtxt(data_file+'test_b_sensor_value.txt')
    sensors_rhs_l = test_rhs_sensor_value[:,sensor_l_index]
    sensors_rhs_r = test_rhs_sensor_value[:,sensor_r_index]
    sensors_b_l = test_b_sensor_value[:,sensor_l_index]
    sensors_b_r = test_b_sensor_value[:,sensor_r_index]


    test_mesh_location = np.loadtxt(data_file+'test_mesh.txt')
    x_index_l = np.where(test_mesh_location <= gamma)[0]
    x_index_r = np.where(test_mesh_location > gamma)[0]   
    x_l = test_mesh_location[x_index_l]
    x_r = test_mesh_location[x_index_r]

    test_u_pre_solution = np.loadtxt(data_file+'test_u_solution.txt')
    label_l = test_u_pre_solution[:,x_index_l]
    label_r = test_u_pre_solution[:,x_index_r]
    
    test_u_grad_pre_solution = np.loadtxt(data_file+'test_u_grad_solution.txt')
    label_grad_l = test_u_grad_pre_solution[:,x_index_l]
    label_grad_r = test_u_grad_pre_solution[:,x_index_r]

 
    test_coef_a_value = np.loadtxt(data_file+'test_coef_a.txt')    
 
    test_boundary = np.loadtxt(data_file+'test_boundary.txt')   

    x_left = []
    x_right =[]
    for i in range(test_rhs_sensor_value.shape[0]):
        x_left.append(np.hstack([
                    np.tile(sensors_rhs_l[i],  (x_l.shape[0], 1)), 
                    np.tile(sensors_rhs_r[i], (x_l.shape[0], 1)), 
                    np.tile(sensors_b_l[i],  (x_l.shape[0], 1)), 
                    np.tile(sensors_b_r[i],  (x_l.shape[0], 1)), 
                    np.tile(test_coef_a_value[0][i],  (x_l.shape[0], 1)), 
                    np.tile(test_coef_a_value[1][i],  (x_l.shape[0], 1)), 
                    np.tile(test_boundary[0][i],  (x_l.shape[0], 1)), 
                    np.tile(test_boundary[1][i],  (x_l.shape[0], 1)),                     
                    x_l[:,None], 
                    label_l[i][:,None],
                    label_grad_l[i][:,None]]))
        

        x_right.append(np.hstack([
                        np.tile(sensors_rhs_l[i],  (x_r.shape[0], 1)), 
                        np.tile(sensors_rhs_r[i], (x_r.shape[0], 1)), 
                        np.tile(sensors_b_l[i],  (x_r.shape[0], 1)), 
                        np.tile(sensors_b_r[i],  (x_r.shape[0], 1)), 
                        np.tile(test_coef_a_value[0][i],  (x_r.shape[0], 1)), 
                        np.tile(test_coef_a_value[1][i],  (x_r.shape[0], 1)), 
                        np.tile(test_boundary[0][i],  (x_r.shape[0], 1)), 
                        np.tile(test_boundary[1][i],  (x_r.shape[0], 1)), 
                            x_r[:,None],
                            label_r[i][:,None],
                            label_grad_r[i][:,None]]))

    x_left =np.vstack(x_left)
    x_right = np.vstack(x_right)


    x_left = (x_left[:, :sensor_num1], 
                   x_left[:,  sensor_num1: sensor_num1+ sensor_num2],
                   x_left[:,  sensor_num1+ sensor_num2: 2*sensor_num1+ sensor_num2],
                   x_left[:,  2*sensor_num1+ sensor_num2: 2*(sensor_num1+ sensor_num2)],
                   x_left[:,-7][:,None],   
                   x_left[:,-6][:,None],                    
                   x_left[:,-5][:,None],  
                   x_left[:,-4][:,None],  
                   x_left[:,-3][:,None],   
                   x_left[:,-2][:,None],   
                   x_left[:,-1][:,None])   
    
    x_right = (x_right[:, :sensor_num1], 
                   x_right[:,  sensor_num1: sensor_num1+ sensor_num2],
                   x_right[:,  sensor_num1+ sensor_num2:2*sensor_num1+ sensor_num2],
                   x_right[:,  2*sensor_num1+ sensor_num2: 2*(sensor_num1+ sensor_num2)],
                   x_right[:,-7][:,None],
                   x_right[:,-6][:,None], 
                   x_right[:,-5][:,None],
                   x_right[:,-4][:,None], 
                   x_right[:,-3][:,None],
                   x_right[:,-2][:,None], 
                   x_right[:,-1][:,None])


    x_left = To_tensor(x_left)
    x_right = To_tensor(x_right)

    return x_left,  x_right, sensor_num1, sensor_num2

X_test_l,  X_test_r, sensor_num1, sensor_num2 = Generate_test_data(data_file, gamma, device)
 
X_test_l = Get_batch(X_test_l,[-3], 10000, device=device)
X_test_r = Get_batch(X_test_r,[-3], 10000, device=device)
 

def Generate_train_data(data_file, gamma,  p):
    boundary_l = np.array([0.])
    boundary_r = np.array([1.])
    gammax = np.array([gamma])

    sensor_location = np.loadtxt(data_file+'sensors_location.txt')
    sensor_l_index = np.where(sensor_location <= gamma)[0]
    sensor_r_index = np.where(sensor_location > gamma)[0]   
    sensor_num1 = len(sensor_l_index)
    sensor_num2 = len(sensor_r_index)      
    

    train_b_sensor_value = np.loadtxt(data_file+'train_b_sensor_value.txt')
    train_rhs_sensor_value = np.loadtxt(data_file+'train_rhs_sensor_value.txt')
    sensors_b_l = train_b_sensor_value[:,sensor_l_index]
    sensors_b_r = train_b_sensor_value[:,sensor_r_index]    
    sensors_rhs_l = train_rhs_sensor_value[:,sensor_l_index]
    sensors_rhs_r = train_rhs_sensor_value[:,sensor_r_index]


    train_mesh_location = np.loadtxt(data_file+'train_mesh.txt')
    x_index_l = np.where(train_mesh_location <= gamma)[0]
    x_index_r = np.where(train_mesh_location > gamma)[0]   
    x_l = train_mesh_location[x_index_l]
    x_r = train_mesh_location[x_index_r]

 
    x_b = np.loadtxt(data_file+'train_b_value.txt') 
    x_b_l = x_b[:,x_index_l]
    x_b_r = x_b[:,x_index_r]       
    x_rhs = np.loadtxt(data_file+'train_rhs_value.txt')
    x_rhs_l = x_rhs[:,x_index_l]
    x_rhs_r = x_rhs[:,x_index_r] 
  

 
    train_coef_a_value = np.loadtxt(data_file+'train_coef_a.txt')   
   
    train_boundary = np.loadtxt(data_file+'train_boundary.txt')  
    
    pp=int(p/2)

    def generate(s1, s2, s3, s4, s5, s6, s7, s8, ix_b_l, ix_b_r, ix_rhs_l, ix_rhs_r):  
        index = random.sample(range(len(x_l)), pp)
        sample_left=np.hstack([
                                np.tile(s1, ( pp, 1)),
                                np.tile(s2, ( pp, 1)), 
                                np.tile(s3, ( pp, 1)),
                                np.tile(s4, ( pp, 1)),
                                np.tile(s5, ( pp, 1)),
                                np.tile(s6, ( pp, 1)),
                                np.tile(s7, ( pp, 1)),
                                np.tile(s8, ( pp, 1)),                
                                x_l[index].reshape(-1,1),
                                ix_b_l[index].reshape(-1,1),
                                ix_rhs_l[index].reshape(-1,1)])

        index = random.sample(range(len(x_r)), pp)
        sample_right = np.hstack([
                                np.tile(s1, ( pp, 1)),
                                np.tile(s2, ( pp, 1)), 
                                np.tile(s3, ( pp, 1)),
                                np.tile(s4, ( pp, 1)),
                                np.tile(s5, ( pp, 1)),
                                np.tile(s6, ( pp, 1)),
                                np.tile(s7, ( pp, 1)),
                                np.tile(s8, ( pp, 1)),
                                x_r[index].reshape(-1,1),
                                ix_b_r[index].reshape(-1,1),
                                ix_rhs_r[index].reshape(-1,1)])
        
        sample_interface = np.hstack([
                                np.tile(s1, ( 1, 1)),
                                np.tile(s2, ( 1, 1)), 
                                np.tile(s3, ( 1, 1)),
                                np.tile(s4, ( 1, 1)),
                                np.tile(s5, ( 1, 1)),
                                np.tile(s6, ( 1, 1)),                                
                                np.tile(s7, ( 1, 1)),
                                np.tile(s8, ( 1, 1)), 
                                gammax.reshape(-1,1)])


        sample_boundaryl = np.hstack([
                                    np.tile(s1, ( 1, 1)),
                                    np.tile(s2, ( 1, 1)), 
                                    np.tile(s3, ( 1, 1)),
                                    np.tile(s4, ( 1, 1)),
                                    np.tile(s5, ( 1, 1)),
                                    np.tile(s6, ( 1, 1)),      
                                    np.tile(s7, ( 1, 1)),
                                    np.tile(s8, ( 1, 1)),  
                                    boundary_l.reshape(-1,1)])
        
        sample_boundaryr = np.hstack([
                                    np.tile(s1, ( 1, 1)),
                                    np.tile(s2, ( 1, 1)), 
                                    np.tile(s3, ( 1, 1)),
                                    np.tile(s4, ( 1, 1)),
                                    np.tile(s5, ( 1, 1)),
                                    np.tile(s6, ( 1, 1)),                                          
                                    np.tile(s7, ( 1, 1)),
                                    np.tile(s8, ( 1, 1)), 
                                    boundary_r.reshape(-1,1)])

        return  (sample_left, sample_right) , sample_interface, (sample_boundaryl, sample_boundaryr)

    sample_left, sample_right =[], []
    sample_interface = []
    sample_boundaryl, sample_boundaryr = [], []

    for i in range(0,train_rhs_sensor_value.shape[0]):      
        if i%200==0:
            print(i)
        sample_domain, sample_gamma, sample_boudnary = generate(sensors_rhs_l[i], sensors_rhs_r[i], sensors_b_l[i], sensors_b_r[i], 
                                                                train_coef_a_value[0][i], train_coef_a_value[1][i],
                                                                train_boundary[0][i], train_boundary[1][i],
                                                                x_b_l[i], x_b_r[i], x_rhs_l[i], x_rhs_r[i])
        sample_left.append(sample_domain[0])
        sample_right.append(sample_domain[1])
        sample_interface.append(sample_gamma)
        sample_boundaryl.append(sample_boudnary[0])
        sample_boundaryr.append(sample_boudnary[1])
 
    sample_left,  sample_right = np.vstack(sample_left), np.vstack(sample_right)
    sample_interface = np.vstack(sample_interface)
    sample_boundaryl, sample_boundaryr = np.vstack(sample_boundaryl), np.vstack(sample_boundaryr)
    
    print(sample_left.shape, sample_right.shape, sample_interface.shape, sample_boundaryl.shape, sample_boundaryr.shape)

    sample_left      =  (sample_left[...,:sensor_num1],  
                         sample_left[...,sensor_num1:sensor_num1+sensor_num2],  
                         sample_left[...,sensor_num1+sensor_num2: 2*sensor_num1+sensor_num2], 
                         sample_left[..., 2*sensor_num1+sensor_num2: 2*(sensor_num1+sensor_num2)],
                         sample_left[...,-7:-6],  
                         sample_left[...,-6:-5], 
                         sample_left[...,-5:-4],  
                         sample_left[...,-4:-3],  
                         sample_left[...,-3:-2], 
                         sample_left[...,-2:-1], 
                         sample_left[...,-1:])   
    sample_right      =  (sample_right[...,:sensor_num1],  
                         sample_right[...,sensor_num1:sensor_num1+sensor_num2],  
                         sample_right[...,sensor_num1+sensor_num2: 2*sensor_num1+sensor_num2], 
                         sample_right[..., 2*sensor_num1+sensor_num2: 2*(sensor_num1+sensor_num2)],
                         sample_right[...,-7:-6], 
                         sample_right[...,-6:-5], 
                         sample_right[...,-5:-4], 
                         sample_right[...,-4:-3], 
                         sample_right[...,-3:-2], 
                         sample_right[...,-2:-1], 
                         sample_right[...,-1:]) 
    
    sample_interface =  (sample_interface[...,:sensor_num1],  
                         sample_interface[...,sensor_num1:sensor_num1+sensor_num2],  
                         sample_interface[...,sensor_num1+sensor_num2: 2*sensor_num1+sensor_num2], 
                         sample_interface[..., 2*sensor_num1+sensor_num2:  2*(sensor_num1+sensor_num2)], 
                         sample_interface[...,-5:-4],  
                         sample_interface[...,-4:-3],  
                         sample_interface[...,-3:-2],  
                         sample_interface[...,-2:-1],  
                         sample_interface[...,-1:])  
     
    sample_boundaryl =  (sample_boundaryl[...,:sensor_num1],  
                         sample_boundaryl[...,sensor_num1:sensor_num1+sensor_num2],  
                         sample_boundaryl[...,sensor_num1+sensor_num2: 2*sensor_num1+sensor_num2], 
                         sample_boundaryl[..., 2*sensor_num1+sensor_num2:  2*(sensor_num1+sensor_num2)], 
                         sample_boundaryl[...,-5:-4],  
                         sample_boundaryl[...,-4:-3],  
                         sample_boundaryl[...,-3:-2], 
                         sample_boundaryl[...,-2:-1],           
                         sample_boundaryl[...,-1:])   
    sample_boundaryr =  (sample_boundaryr[...,:sensor_num1],  
                         sample_boundaryr[...,sensor_num1:sensor_num1+sensor_num2],  
                         sample_boundaryr[...,sensor_num1+sensor_num2: 2*sensor_num1+sensor_num2], 
                         sample_boundaryr[..., 2*sensor_num1+sensor_num2:  2*(sensor_num1+sensor_num2)], 
                         sample_boundaryr[...,-5:-4],  
                         sample_boundaryr[...,-4:-3],  
                         sample_boundaryr[...,-3:-2], 
                         sample_boundaryr[...,-2:-1],  
                         sample_boundaryr[...,-1:])   

    x_train_l = To_tensor(sample_left)
    x_train_r = To_tensor(sample_right)

    ix_train = To_tensor(sample_interface)

    bx_train_l = To_tensor(sample_boundaryl)
    bx_train_r = To_tensor(sample_boundaryr)
 
    return (x_train_l,  x_train_r),  ix_train,  (bx_train_l,bx_train_r)

sample_domain, sample_gamma, sample_boundary = Generate_train_data(data_file, gamma , p=p)

 
model = IONet(sensor_num1, sensor_num2, sensor_num1, sensor_num2, 1, 1, 1, 1, width=width, actv=nn.Tanh()).to(device)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
 
optimizer=optim.Adam(model.parameters(), lr=lr)
t0=time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    input_left = Get_batch(sample_domain[0], requires_grad_remark=[-3], batch_size=batch_size, device=device)
    u1= model(input_left,label='l')
    grad1=grad(u1,input_left[-3])
    laplace1=-grad(grad1,input_left[-3])*input_left[4]+input_left[-2]*u1         
    loss_pde_residual1 = nn.MSELoss()(laplace1, input_left[-1])

    input_right = Get_batch(sample_domain[1], requires_grad_remark=[-3], batch_size=batch_size, device=device)
    u2= model(input_right,label='r')
    grad2=grad(u2,input_right[-3])
    laplace2=-grad(grad2,input_right[-3])*input_right[5] + input_right[-2]*u2      
    loss_pde_residual2 = nn.MSELoss()(laplace2, input_right[-1])

   
    input_gamma = Get_batch(sample_gamma, requires_grad_remark=[-1], batch_size=batch_size, device=device)
    ui_1l = model(input_gamma,label='l')
    ui_1r = model(input_gamma,label='r')
    loss_interface_d = nn.MSELoss()(ui_1l,ui_1r)     ### 1

    z=torch.ones(input_gamma[0].size()[0]).view(-1,1).to(input_gamma[0].device)
    grad_i11=torch.autograd.grad(ui_1l,input_gamma[-1], grad_outputs=z, create_graph=True)[0]
    grad_i12=torch.autograd.grad(ui_1r,input_gamma[-1], grad_outputs=z, create_graph=True)[0] 
    loss_interface_n=nn.MSELoss()(grad_i11*input_gamma[4], grad_i12*input_gamma[5])
    
   
    input_boundaryl =  Get_batch(sample_boundary[0], requires_grad_remark=[], batch_size=batch_size, device=device)
    loss_bd_l = nn.MSELoss()(model(input_boundaryl,label='l'), input_boundaryl[-3]) # 1
    input_boundaryr =  Get_batch(sample_boundary[1], requires_grad_remark=[], batch_size=batch_size, device=device)
    loss_bd_r = nn.MSELoss()(model(input_boundaryr,label='r'), input_boundaryr[-2]) 

    loss = loss_pde_residual1+loss_pde_residual2+ 10*(loss_interface_d+loss_interface_n) + 100*(loss_bd_l+loss_bd_r)
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch%print_epoch==0:
        print('epoch {}: training loss'.format(epoch), loss.item(),optimizer.param_groups[0]['lr'])
        print(loss_pde_residual1.item(), loss_pde_residual2.item(), 
              loss_interface_d.item(),loss_interface_n.item(),loss_bd_l.item(),loss_bd_r.item())
        pre_label_l = model(X_test_l,label='l')
        pre_label_r = model(X_test_r,label='r')
        relative_l2 =((X_test_l[-2]-pre_label_l)**2).sum()+((pre_label_r-X_test_r[-2])**2).sum()
        relative_l2 = torch.sqrt(relative_l2/(((X_test_l[-2])**2).sum()+((X_test_r[-2])**2).sum()))

 
        left_grad_pre = grad(pre_label_l,X_test_l[-3])
        right_grad_pre = grad(pre_label_r,X_test_r[-3])
        relative_l2_grad = ((X_test_l[-1]-left_grad_pre)**2).sum()+((X_test_r[-1]-right_grad_pre)**2).sum()
        relative_l2_grad = torch.sqrt(relative_l2_grad/(((X_test_l[-1])**2).sum()+((X_test_r[-1])**2).sum()))        

        print('Rela L2 loss is: ', relative_l2.item(), relative_l2_grad.item())
        print('\n')


    if (epoch+1)%int(epochs/100)==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

print('time',time.time()-t0)
if not os.path.isdir('./'+'model/'): os.makedirs('./'+'model/')
torch.save(model,'model/'+'pi_ion'+'_'+str(width)+'_'+str(epoch)+'.pkl')


