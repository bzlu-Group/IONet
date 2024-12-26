# -*- coding: utf-8 -*-

import numpy as np 
import torch
import torch.nn as nn
from functools import wraps
import random

def grad(y,x): 
    dydx, = torch.autograd.grad(outputs=y,inputs=x,retain_graph=True,grad_outputs=torch.ones(y.size()).to(y.device) ,
                                create_graph=True,allow_unused=True)
    return dydx


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

def data_transform(data):

    x=data[0].view(-1,1)
    y=data[1].view(-1,1)
    z=data[2].view(-1,1)   
    x=x.clone().detach().requires_grad_(True)
    y=y.clone().detach().requires_grad_(True)
    z=z.clone().detach().requires_grad_(True)
    input=torch.cat((x,y,z),1)

    return x,y,z,input

def gradient(data,x,y,z):
    dx=grad(data,x)
    dy=grad(data,y)
    dz=grad(data,z)
    
    return dx,dy,dz


def Get_batch(X, requires_grad_remark, batch_size, device):
    """
    inputs:
        X: a tuple. [sensors, x, label]
        requires_grad_remark: remarks. [2]: means that X[2].requires_grad_(True)
    """
    out= []
    if batch_size ==[]:
        for item in X:
            out.append(item.to(device))
        for i in requires_grad_remark:
            out[i] = out[i].requires_grad_(True)
    else:
        index = random.sample(range(X[0].shape[0]), batch_size)
        for item in X:
            out.append(item[index].to(device))
        for i in requires_grad_remark:
            out[i] = out[i].requires_grad_(True)

    return out
