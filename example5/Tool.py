import numpy as np
import torch
import csv
from functools import wraps

def data_transform(data,device): 
    x=data[0].view(-1,1)
    y=data[1].view(-1,1)
    x=x.clone().detach().requires_grad_(True)
    y=y.clone().detach().requires_grad_(True)
    input=torch.cat((x,y),1)

    return x,y,input


def gradient(data,x,y,device):
    dx=grad(data,x,device)
    dy=grad(data,y,device)

    return dx,dy


def Grad(y, x, create_graph=True, keepdim=False):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Nx] or [Nx]
    Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
    '''
    N = y.size(0) if len(y.size()) == 2 else 1
    Ny = y.size(-1)
    Nx = x.size(-1)
    z = torch.ones_like(y[..., 0])
    dy = []
    for i in range(Ny):
        dy.append(torch.autograd.grad(y[..., i], x, grad_outputs=z, create_graph=create_graph)[0])
    shape = np.array([N, Ny])[2-len(y.size()):]
    shape = list(shape) if keepdim else list(shape[shape > 1])
    return torch.cat(dy, dim=-1).view(shape + [Nx])


def grad(y,x,device):
    '''return tensor([dfdx,dfdy,dfdz])
    '''    
    dydx, = torch.autograd.grad(outputs=y,inputs=x,retain_graph=True,grad_outputs=torch.ones(y.size()),
                                create_graph=True,allow_unused=True)
    return dydx


def map_elementwise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        container, idx = None, None
        for arg in args:
            if type(arg) in (list, tuple, dict):
                container, idx = type(arg), arg.keys() if type(arg) == dict else len(arg)
                break
        if container is None:
            for value in kwargs.values():
                if type(value) in (list, tuple, dict):
                    container, idx = type(value), value.keys() if type(value) == dict else len(value)
                    break
        if container is None:
            return func(*args, **kwargs)
        elif container in (list, tuple):
            get = lambda element, i: element[i] if type(element) is container else element
            return container(wrapper(*[get(arg, i) for arg in args], 
                                     **{key:get(value, i) for key, value in kwargs.items()}) 
                             for i in range(idx))
        elif container is dict:
            get = lambda element, key: element[key] if type(element) is dict else element
            return {key:wrapper(*[get(arg, key) for arg in args], 
                                **{key_:get(value_, key) for key_, value_ in kwargs.items()}) 
                    for key in idx}
    return wrapper

def div(y, x, create_graph=True, keepdim=False):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Ny] or [Ny]
    Return dy/dx ([N, 1] or [1]).
    '''
    if x.size(-1) !=  y.size(-1): raise ValueError
    grady = Grad(y, x, create_graph=create_graph, keepdim=keepdim)
    p = torch.eye(x.size(-1), dtype = x.dtype,device=x.device).view(x.size(-1)**2,1)

    return grady.view(-1,x.size(-1)**2)@p


def read_pqr(file_name,device):
    """
    input: pqr file,d (d is a parameter decide training interface loaction!)
    return: centers,rs,qs  [datype: array]
    """
    f=open(file_name)
    line=f.readline()
    centers=[]
    rs=[]
    qs=[]
    while line:
        centers.append(list(map(float,line.split()[5:8])))
        qs.append(list(map(float,line.split()[8:9]))[0])
        rs.append(list(map(float,line.split()[9:]))[0])
        line=f.readline()
    centers=torch.tensor(centers).float().to(device)
    rs=(torch.tensor(rs)).float().to(device)
    qs=torch.tensor(qs).float().to(device)
    return centers,rs,qs

def read_csv(csv_file,device):
    """
    H,u,G,point1,point2,point3
    """
    data=[]
    with open(csv_file) as file:
        reader = csv.reader(file)
        head = next(reader)
        for row in reader:
            data.append(list(map(float,row)))
    data=torch.tensor(data).float().to(device)
    return data

def read_xyzfile(xyz_file,device):
    """
    read :  mesh file ,pqrfile
    return :  inner bound point and faxiang
    """
    f=open(xyz_file)
    line=f.readline()
    fdirection=[]
    inner_b=[]
    while line:
        # Vertices : all point       
        if len(line.split())==6 and ( list(map(float,line.split()))[0:3] not in inner_b):
            inner_b.append(list(map(float,line.split()))[0:3])
            fdirection.append(list(map(float,line.split()))[3:])                                               
        else:
            pass
        line=f.readline()
    # get inner point    
    inner_b=torch.tensor(inner_b).float().to(device)
    fdirection=torch.tensor(fdirection).float().to(device)
    return inner_b, fdirection

def data_label_mesh(mesh_file,device):
    f=open(mesh_file)
    line=f.readline()
    All_point=[]
    inner_point_index=[]
    out_point_index=[]
    inner_b_index=[]
    out_b_index=[]
    k=0
    
    interface=[] 
    
    while line:
        # Vertices : all point       
        if line.split()[0]=='Vertices' :
            k=1
        elif line.split()[0]=='Tetrahedra':
            k=2
        elif line.split()[0]=='Triangles':
            k=3
        elif line.split()[0]=='END':
            k=0
                
        elif k==1:
            All_point.append(list(map(float,line.split()))[0:3])
        
        elif k==2:
            if list(map(float,line.split()))[-1]==2:
                out_point_index=list(set(out_point_index).union(set(list(map(int,line.split()))[0:4])))
            elif list(map(float,line.split()))[-1]==1:
                inner_point_index=list(set(inner_point_index).union(set(list(map(int,line.split()))[0:4])))          
        elif k==3:
            if list(map(float,line.split()))[-1]==9:
                inner_b_index=list(set(inner_b_index).union(set(list(map(int,line.split()))[0:3])))
                interface.append(list(map(int,line.split()))[0:3])
                
            elif list(map(float,line.split()))[-1]==5:
                out_b_index=list(set(out_b_index).union(set(list(map(int,line.split()))[0:3])))          
            else:
                k=0
        line=f.readline()
    
    All_point=torch.tensor(All_point)
    
    inner_point_index=list(set(inner_point_index)-set(inner_b_index))
    out_point_index=list(set(out_point_index)-set(inner_b_index+out_b_index)) 

    out_point_index=torch.tensor(out_point_index)-1
    inner_point_index=torch.tensor(inner_point_index)-1 
    boundary_index=torch.tensor(out_b_index)-1

    inner_point=(All_point[inner_point_index]).clone().detach()
    out_point=All_point[out_point_index].clone().detach()
    boundary_point=All_point[boundary_index].clone().detach()

     
    return inner_point.to(device),out_point.to(device), boundary_point.to(device), inner_point_index, out_point_index,boundary_index


def mesh_data_label(mesh_file,csv_file,device):
    # G h u_r
    inner_point,out_point, boundary_point, inner_point_index, out_point_index,boundary_index=data_label_mesh(mesh_file,device)
    data= read_csv(csv_file,device)
    inner_label=data[inner_point_index,0:1] + data[inner_point_index,2:3] + data[inner_point_index,1:2]  
    G = data[inner_point_index,1:2]
    out_label=data[out_point_index,0:1]      
    boundary_label=data[boundary_index,0:1]  
    return inner_point,inner_label,G ,out_point,out_label, boundary_point, boundary_label
