import numpy as np 
import argparse
import torch
import itertools
import torch.optim as optim
from Tool import  Grad, div, map_elementwise
from Net_type import  IONet
from Generate_data import Data
import os
alpha=7.0465*1e3
pi=torch.tensor(np.pi)
Is=1e-5 



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

    data=Data(args.mesh_file, args.pqr_file, args.xyz_file, args.csv_file, device=device)
    # train data 
    TX_train_in, TX_train_out, TX_train_interface, TX_train_b = data.Generate_train_data(args.train_sample_num)
    batch_function = batch_type(args.batch)
    input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)

    model=IONet(sensor_in=1,  sensor_out=1,  m=args.unit).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')


    loss_history =[]
    for epoch in range(args.nepochs):      
        optimizer.zero_grad()    
        U1=model(input_in,label='inner')
        grads_in= Grad(U1,input_in[2])
        div_in = -div(grads_in,input_in[2])*input_in[0]
        loss_in=torch.mean((div_in)**2)

        U1_b = model(input_gamma , label='inner')
        U2_b = model(input_gamma , label='out')        
        loss_gammad=torch.mean((U2_b-U1_b-input_gamma[-2])**2)

        dU1_N=torch.autograd.grad(U1_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        G_NN=((input_gamma[1]*dU2_N-input_gamma[0]*(dU1_N+input_gamma[-1]))*input_gamma[-3]).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN)**2)   
        
        k2 = 8.4869*Is
        U2 = model(input_out,  label='out')
        grads_out = Grad(U2,input_out[2])
        div_out = -input_out[1]*div(grads_out,input_out[2])+k2*torch.sinh(U2)
        loss_out=torch.mean((div_out)**2)
        
        loss_boundary=torch.mean((model(input_boundary,  label='out')-input_boundary[-1])**2)

        loss = loss_in + loss_out + loss_gamman + loss_gammad + 100*loss_boundary
        loss.backward(retain_graph=True)
        optimizer.step()

        input_in, input_out, input_gamma, input_boundary = batch_function(args, TX_train_in, TX_train_out, TX_train_interface, TX_train_b, device)

                
        if (epoch+1)%args.print_num==0:
            with torch.no_grad():    
                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      
                print('Epoch,  Training MSE: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
                print('        loss_in {}, loss_out {}, loss_gamman {}, loss_gammad {}, loss_boundary {}'.format(loss_in.item(),loss_out.item(),loss_gamman.item(),loss_gammad.item(),loss_boundary.item()))
                
                if epoch>args.nepochs*0.98:             
                    torch.save(model, 'outputs/'+args.filename+'/model/{}model.pkl'.format(epoch))

        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--unit', type=int, default=150)
   
    parser.add_argument('--train_sample_num', type=int, default=1000)  
    parser.add_argument('--batch',type=str, default='batch') 
    parser.add_argument('--btrain_inner', type=int, default=2000)
    parser.add_argument('--btrain_out', type=int, default=2000)  
    parser.add_argument('--btrain_gamma', type=int, default=2000)
    parser.add_argument('--btrain_boundary', type=int, default=2000)

    parser.add_argument('--mesh_file', type=str, default='ADP/ADP.mesh') 
    parser.add_argument('--pqr_file', type=str, default='ADP/transfer_ADP.pqr') 
    parser.add_argument('--xyz_file', type=str, default='ADP/ADP.xyz') 
    parser.add_argument('--csv_file', type=str, default='ADP/ADP.csv')     
 
    parser.add_argument('--print_num', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.001)       
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)

           
