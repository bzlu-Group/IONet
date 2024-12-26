import numpy as np 
import argparse
import torch
import torch.nn as nn
import time
import itertools
import random
import torch.optim as optim
from Tool import  div,Grad,  Get_batch
from Net_type import IONet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from GenerateData import Data
torch.set_default_dtype(torch.float32)



def main(args):
    if torch.cuda.is_available and args.cuda:
        device='cuda:1'
        print('cuda is avaliable')
    else:
        device='cpu'

    data = Data(gap=args.gap)
    # train data
    TX_train_up, TX_train_down, TX_train_interface, TX_train_b_up, TX_train_b_down = data.Generate_train_data(args.sample_num)
    TX_train_up, TX_train_down = Get_batch(TX_train_up, [], [], device),Get_batch(TX_train_down, [], [], device)
    TX_train_interface, TX_train_b_up, TX_train_b_down = Get_batch(TX_train_interface,[],[],device), Get_batch(TX_train_b_up, [], [], device), Get_batch(TX_train_b_down, [], [], device)
   
    # test data
    x_test_up,  x_test_down = data.Generate_test_data()
    x_test_up,  x_test_down = Get_batch(x_test_up, [], [], device), Get_batch(x_test_down, [], [], device)

    sensor_num_up = data.sensors_nup
    sensor_num_down = data.sensors_ndown
    sensor_num_whole = sensor_num_down+sensor_num_up


    model=IONet(sensor_num_up, sensor_num_down, 1,1, width=args.width).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    t0=time.time()
    batch_size = 1000
    if not os.path.isdir('./model'): os.makedirs('./model')

    for epoch in range(args.nepochs):
        optimizer.zero_grad()
     
        input_up =  Get_batch(TX_train_up, [4], batch_size, device)
        U1=model(input_up,label='up')
        grads_in=Grad(U1,input_up[4])
        div_in = -div(grads_in,input_up[4]) * input_up[3]
        loss_up=torch.mean((div_in)**2)  
 
        input_interface =  Get_batch(TX_train_interface, [4], batch_size, device)
        U1_b=model(input_interface,label='up')
        U2_b_in=model(input_interface,label='down')          
        loss_gammad=torch.mean((U1_b-U2_b_in)**2)

        dU1_N=torch.autograd.grad(U1_b,input_interface[4], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]      
        U1_N=dU1_N[:,1].view(-1,1) * input_interface[3]   
        dU2_N=torch.autograd.grad(U2_b_in,input_interface[4], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        U2_N=dU2_N[:,1].view(-1,1) * input_interface[2]       
        loss_gamman=torch.mean((U1_N-U2_N)**2)   

 
        input_down = Get_batch(TX_train_down, [4], batch_size, device)
        U2 = model(input_down,label='down') 
        grads_out=Grad(U2,input_down[4])
        div_out = -div(grads_out,input_down[4])* input_down[2]          
        loss_down=torch.mean((div_out)**2)
        
        input_boundary_up=Get_batch(TX_train_b_up, [],  batch_size, device)
        loss_boundary_up=torch.mean((model(input_boundary_up,label='up')-input_boundary_up[-1])**2)
        input_boundary_down=Get_batch(TX_train_b_down, [],  batch_size, device)
        loss_boundary_down=torch.mean((model(input_boundary_down,label='down')-input_boundary_down[-1])**2)

        loss = loss_up+ loss_down + 10*(loss_gammad +loss_gamman) + 100*(loss_boundary_up + loss_boundary_down)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if (epoch+1)%args.print_num==0:
 
            with torch.no_grad():

                Mse_train=(loss_up+loss_down + loss_gammad +loss_gamman + loss_boundary_up + loss_boundary_down).item()      
                print('Epoch,  Training MSE          : ',epoch+1,Mse_train, optimizer.param_groups[0]['lr'])
                print('        loss up/down          : ',loss_up.item(), loss_down.item())
                print('        loss interd/intern    : ',loss_gammad.item(), loss_gamman.item())
                print('        loss boundup/bounddown: ',loss_boundary_up.item(), loss_boundary_down.item())
                
                test_label = model(x_test_up,label='up').view(100,-1)
                label_up = x_test_up[-1].view(100,-1)
                L2_up_loss = torch.sqrt(((test_label-label_up)**2).sum(dim=-1)/((label_up)**2).sum(dim=-1))
       
                test_label = model(x_test_down,label='down').view(100,-1)
                label_down = x_test_down[-1].view(100,-1)
                L2_down_loss = torch.sqrt(((test_label-label_down)**2).sum(dim=-1)/((label_down)**2).sum(dim=-1))
           
                
                print('Test Losses')
                print('        Relative L2 up(mean/std)   :',torch.mean(L2_up_loss).item(), torch.std(L2_up_loss).item())  
                print('        Relative L2 down(mean/std) :',torch.mean(L2_down_loss).item(), torch.std(L2_down_loss).item())                                                                                 
                print('*****************************************************')
        
                   
        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95
            
    print('totle use time:',time.time()-t0)
    torch.save(model, './model/gap_{}_model_sensor{}_sample_{}.pkl'.format(args.gap, sensor_num_whole, args.sample_num))    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='result')
    parser.add_argument('--width', type=int, default = 120)
    parser.add_argument('--gap', type=int, default = 8) 
    parser.add_argument('--print_num', type=int, default = 1000)
    parser.add_argument('--nepochs', type=int, default = 100000)   
    parser.add_argument('--sample_num', type=int, default = 500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--save', type=str, default=False)

    args = parser.parse_args()
    main(args)

        
            
