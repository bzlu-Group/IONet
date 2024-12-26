import numpy as np 
import pyDOE
import argparse
import torch
import time,os
import itertools
import torch.optim as optim
from Tool import  Grad, div, map_elementwise
from Net_type import  ION_Stacked
from GenerateData import Data

def test_exact_solution(X,label):

    if label == 'inner':
        solution  = torch.sin(X[:,0:1])*torch.sin(X[:,1:2])*torch.sin(X[:,2:3])*torch.sin(X[:,3:4])*torch.sin(X[:,4:5])*torch.sin(X[:,5:6])

    elif label == 'out':
        solution =  torch.exp(X[:,0:1]+X[:,1:2]+X[:,2:3]+X[:,3:4]+X[:,4:5]+X[:,5:6])
    else:
        raise ValueError("invalid label for u(x)")
   
    return solution.view(-1,1)


def get_batch(args, out, inner, gamma, input_boundary, device):
    # batch
    @map_elementwise
    def batch_mask(X, num):
        return np.random.choice(X.size(0), num, replace=False)
    @map_elementwise
    def batch(X, mask):
        return X[mask].requires_grad_(True).float().to(device)

    mask1 = batch_mask(out[-2], args.btrain_out)
    mask2 = batch_mask(inner[-2], args.btrain_inner)
    mask3 = batch_mask(gamma[-2], args.btrain_gamma)
    mask4 = batch_mask(input_boundary[-2], args.btrain_boundary)

    return batch(out, mask1),  batch(inner, mask2), batch(gamma, mask3), batch(input_boundary, mask4), 


def get_full_batch(args, out, inner, gamma, input_boundary_in, device):
    # batch
  
    @map_elementwise
    def batch(X):
        return X.requires_grad_(True).float().to(device)

    return batch(out),  batch(inner), batch(gamma), batch(input_boundary_in)


def batch_type(batch, get_batch=get_batch, get_full_batch=get_full_batch):
    if batch=='full':
        return get_full_batch
    else:
        return get_batch



def main(args):

    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'
    
    ### train data
    data=Data( num_train_sample=args.num_train_sample,sensor_num=args.sensor_num)
    
    # test data
    test_inner, label_inner, test_out, label_out = data.Generate_test_data(device)
    
    # train data 
    TX_train_in, TX_train_out, TX_train_interface, TX_train_b = data.Generate_train_data()
    batch_function = batch_type(args.batch)
    input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)

    model=ION_Stacked(sensor_in=data.num_sensor_in,  sensor_out=data.num_sensor_out,  m=args.inner_unit).to(device) 
    
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    t0=time.time()
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')
    

    for epoch in range(args.nepochs):     
        optimizer.zero_grad()    
        U1=model(input_in,label='inner')
        grads_in= Grad(U1,input_in[2])
        div_in = -div(grads_in,input_in[2])
        loss_in=torch.mean((div_in-input_in[-1])**2)

        U1_b=model(input_gamma,label='inner')
        U2_b_in=model(input_gamma,label='out')      
        loss_gammad=torch.mean((U2_b_in-U1_b-input_gamma[-2])**2)

        dU1_N=torch.autograd.grad(U1_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b_in,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        G_NN=((1e-3*dU2_N-dU1_N)*input_gamma[-3]/0.6).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN-input_gamma[-1])**2)   
        
        U2 =model(input_out,label='out')
        grads_out = Grad(U2,input_out[2])
        div_out = -1e-3*div(grads_out,input_out[2])      
        loss_out=torch.mean((div_out-1e-3*input_out[-1])**2)

        loss_boundary=torch.mean((model(input_boundary,label='out')-input_boundary[-1])**2)

        loss = loss_in+loss_out+loss_gamman+loss_gammad+ 100*loss_boundary
        loss.backward(retain_graph=True)
        optimizer.step()

        input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)
               
        if (epoch+1)%args.print_num==0:
             
            with torch.no_grad():    

                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      

                print('Epoch: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
                print('       ', loss_in.item(),loss_out.item(),loss_gamman.item(),loss_gammad.item(),loss_boundary.item())
                # L_infty
                L_inf_inner_loss=torch.max(torch.abs(model(test_inner,label='inner')-label_inner))
                L_inf_out_loss=torch.max(torch.abs(model(test_out,label='out')-label_out))
                print('L_infty:',max(L_inf_inner_loss.item(),L_inf_out_loss.item()))
                # relative l2 error
                aa=((model(test_inner,label='inner')-label_inner)**2).sum()
                bb=((model(test_out,label='out')-label_out)**2).sum()
                cc=((label_inner)**2).sum()
                dd=((label_out)**2).sum()
                print('Relative L2 error:',torch.sqrt((aa+bb)/(cc+dd)).item())
                print('************************************')

                if args.save and epoch>args.nepochs*0.98:             
                    torch.save(model, 'outputs/'+args.filename+'/model/{}.pkl'.format(epoch))

        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

               
    print('Totle training time:',time.time()-t0)
        
if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--inner_unit', type=int, default=50)
    parser.add_argument('--out_unit', type=int, default=50)   
    parser.add_argument('--num_train_sample', type=int, default=100)  
    parser.add_argument('--sensor_num', type=int, default=20)


    # batch of training data
    parser.add_argument('--batch',type=str, default='batch') # full: full batch else batch size!
    parser.add_argument('--btrain_inner', type=int, default=2000)
    parser.add_argument('--btrain_out', type=int, default=2000)  
    parser.add_argument('--btrain_gamma', type=int, default=2000)
    parser.add_argument('--btrain_boundary', type=int, default=2000) 

    parser.add_argument('--print_num', type=int, default=200)
    parser.add_argument('--nepochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.001)           
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--box', type=list, default=[-1,1,-1,1])
    parser.add_argument('--change_epoch', type=int, default=4000)
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)

           
