import numpy as np 
import argparse
import torch
import time,os
import itertools
import torch.optim as optim
from Tool import  Grad, div,  Get_batch
from Net_type import  IONet
from GenerateData import Data


def u(x,label):
    """
    exact solution
    """
    x=x.t()  
    if label=='inner':
        u=(1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    elif label=='out':
        u=(2/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    else:
        raise ValueError("invalid label for u(x)")
   
    return u



def inter_dirich(x):
    x=x.t()
    return (1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)


def test_data_net(args, device):  
    
    step=0.02
    x = np.arange(-1, 1+step, step)
    y = np.arange(-1, 1+step, step)
    xx,yy=np.meshgrid(x,y)
    input_x=torch.tensor(xx).view(-1,1).to(device)
    input_y=torch.tensor(yy).view(-1,1).to(device)
    input=(torch.cat((input_x,input_y),1)).float()
    x=input[:,0]
    y=input[:,1]

    rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
    rr=torch.tensor(rr) 
    r=0.65**(2/3)
    location=torch.where(rr<r)[0]
    test_inner=(input[location,:])
    location=torch.where(rr>r)[0]
    test_out=(input[location,:])
    label_out=u(test_out,label='out')
    label_inner=u(test_inner,label='inner')

    return test_out.to(device),label_out.to(device),test_inner.to(device),label_inner.to(device)




def main(args):

    if torch.cuda.is_available and args.cuda:
        device='cuda:0'
        print('cuda is avaliable')
    else:
        device='cpu'

    ### train data
    data=Data(args.step, args.sample_num, device=device)

    # test data
    test_inner, label_inner, test_out, label_out = data.Generate_test_data()

    # train data 
    TX_train_in, TX_train_out, TX_train_interface, TX_train_b = data.Generate_train_data()

    model=IONet(sensor_in=data.num_sensor_in, sensor_out=data.num_sensor_out, width=args.width, layers=args.depth).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer=optim.Adam(itertools.chain(model.parameters()),lr=args.lr)
    t0=time.time()

    loss_history =[]
    batch_size = 1000
    for epoch in range(args.nepochs):      
        optimizer.zero_grad()    
        input_in = Get_batch(TX_train_in, [2], batch_size, device)
        U1=model(input_in,label='inner')
        grads_in= Grad(U1,input_in[2])
        div_in = -div(grads_in,input_in[2])
        loss_in=torch.mean((2*div_in-input_in[-1])**2)

        input_gamma = Get_batch(TX_train_interface, [2], batch_size, device)
        U1_b = model(input_gamma , label='inner')
        U2_b = model(input_gamma , label='out')        
        loss_gammad=torch.mean((U2_b-U1_b-inter_dirich(input_gamma[2]))**2)

        dU1_N=torch.autograd.grad(U1_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b,input_gamma[2], grad_outputs=torch.ones_like(U1_b).to(device), create_graph=True)[0]
        G_NN=((dU2_N-2*dU1_N)*input_gamma[-1]).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN)**2)   
        
        input_out = Get_batch(TX_train_out, [2], batch_size, device)
        U2 = model(input_out,  label='out')
        grads_out = Grad(U2,input_out[2])
        div_out = -div(grads_out,input_out[2])      
        loss_out=torch.mean((div_out-input_out[-1])**2)

        input_boundary = Get_batch(TX_train_b, [], batch_size, device)
        loss_boundary=torch.mean((model(input_boundary,  label='out')-input_boundary[-1])**2)

        loss = loss_in + loss_out + 10*(loss_gamman + loss_gammad) + 100*loss_boundary
        loss.backward(retain_graph=True)
        optimizer.step()

   
        if (epoch+1)%args.print_num==0:
            with torch.no_grad():    
                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      
                print('Epoch,  Training MSE: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
                print(loss_in.item(), loss_out.item(), loss_gamman.item() , loss_gammad.item() ,loss_boundary.item())
                # rela L_2
                l2_error = ((model(test_inner,label='inner')-label_inner)**2).sum()+((model(test_out,label='out')-label_out)**2).sum()
                l2_error =  torch.sqrt(l2_error/(((label_inner)**2).sum()+((label_out)**2).sum()))

                # L_infty
                L_inf_inner_loss=torch.max(torch.abs(model(test_inner,label='inner')-label_inner))
                L_inf_out_loss=torch.max(torch.abs(model(test_out,label='out')-label_out))
                print('L_infty:',max(L_inf_inner_loss.item(),L_inf_out_loss.item()))
                print('Rel. L_2:',l2_error.item())
                print('*****************************************************')  

                if args.save:
                    loss_history.append([epoch,loss_in.item(),loss_gammad.item(),loss_out.item(),loss_boundary.item(),loss_gamman.item()])
                    loss_record = np.array(loss_history)
                    np.savetxt('outputs/'+args.filename+'/loss_record.txt', loss_record)           

        if  (epoch+1)%int(args.nepochs/100)==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.95

    if not os.path.isdir('./model/'): os.makedirs('./model/')
    torch.save(model,'./model/model_width_{}_depth_{}_samples_{}_sensor_{}_{}.pkl'.format(args.width, args.depth, args.sample_num,data.num_sensor_in, data.num_sensor_out))
    print('Totle training time:',time.time()-t0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--sample_num', type=int, default=320)
    parser.add_argument('--step', type=int, default = 10 ) # [3, 5, 9, 10]
    parser.add_argument('--print_num', type=int, default = 200)
    parser.add_argument('--nepochs', type=int, default = 40000)
    parser.add_argument('--lr', type=float, default = 0.001)       
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--box', type=list, default=[-1,1,-1,1])
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)

           
