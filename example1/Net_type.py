import torch.nn as nn
import torch

class IONet(nn.Module):
    def __init__(self, sensor_num1,sensor_num2, m=20, actv=nn.Tanh()):
        super(IONet, self).__init__()
        self.actv=actv
        self.b1linear_input = nn.Linear(sensor_num1, m)    
        self.b1linear1 =nn.Linear(m,m)
        self.b1linear2 = nn.Linear(m,m)
        self.b1linear3 =nn.Linear(m,m)
        self.b1linear4 = nn.Linear(m,m)

        self.b2linear_input = nn.Linear(sensor_num2, m)    
        self.b2linear1 =nn.Linear(m,m)
        self.b2linear2 = nn.Linear(m,m)
        self.b2linear3 =nn.Linear(m,m)
        self.b2linear4 = nn.Linear(m,m)

        self.t1linear_input = nn.Linear(1, m)    
        self.t1linear1 =nn.Linear(m,m)
        self.t1linear2 = nn.Linear(m,m)
        self.t1linear3 =nn.Linear(m,m)
        self.t1linear4 = nn.Linear(m,m)

        self.t2linear_input = nn.Linear(1, m)    
        self.t2linear1 =nn.Linear(m,m)
        self.t2linear2 = nn.Linear(m,m)
        self.t2linear3 =nn.Linear(m,m)
        self.t2linear4 = nn.Linear(m,m)

        self.p = self.__init_params()


    def forward(self, X, label):
        feature_l = X[0]
        y = self.actv(self.b1linear_input(feature_l))
        y = self.actv(self.b1linear2(self.actv(self.b1linear1(y))))
        branch1 =  self.b1linear4(self.actv(self.b1linear3(y)))
        
        feature_m = X[1]
        y = self.actv(self.b2linear_input(feature_m))
        y = self.actv(self.b2linear2(self.actv(self.b2linear1(y))))
        branch2 = self.b2linear4(self.actv(self.b2linear3(y)))
    
        x = X[2]
        if label == "l":  
            y = self.actv(self.t1linear_input(x))
            y = self.actv(self.t1linear2(self.actv(self.t1linear1(y))))
            truck = self.t1linear4(self.actv(self.t1linear3(y)))
            output = torch.sum(branch1*branch2*truck, dim=-1, keepdim=True) + self.p['bias1']

        elif label == 'r':
            y = self.actv(self.t2linear_input(x))
            y = self.actv(self.t2linear2(self.actv(self.t2linear1(y))))
            truck = self.t2linear4(self.actv(self.t2linear3(y)))  
            output = torch.sum(branch1*branch2*truck, dim=-1, keepdim=True) + self.p['bias2']   

        return output

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias1'] = nn.Parameter(torch.zeros([1]))
        params['bias2'] = nn.Parameter(torch.zeros([1]))
        return params


class DeepONet(nn.Module):
    def __init__(self, sensor_num, m=20,actv=nn.ReLU()):
        super(DeepONet, self).__init__()
        self.actv=actv
        self.b1linear_input = nn.Linear(sensor_num,m)    
        self.b1linear1 =nn.Linear(m,m)
        self.b1linear2 = nn.Linear(m,m)
        self.b1linear3 =nn.Linear(m,m)
        self.b1linear4 = nn.Linear(m,m)

        self.t1linear_input = nn.Linear(1,m)    
        self.t1linear1 =nn.Linear(m,m)
        self.t1linear2 = nn.Linear(m,m)
        self.t1linear3 =nn.Linear(m,m)
        self.t1linear4 = nn.Linear(m,m)

        self.p = self.__init_params()
 
      
    def forward(self, X, tol=0):
        feature_up = X[0]
        y = self.actv(self.b1linear_input(feature_up))
        y = self.actv(self.b1linear2(self.actv(self.b1linear1(y))))
        branch1 =  self.b1linear4(self.actv(self.b1linear3(y)))

        x = X[1]
        y = self.actv(self.t1linear_input(x+tol))
        y = self.actv(self.t1linear2(self.actv(self.t1linear1(y))))
        truck = self.t1linear4(self.actv(self.t1linear3(y)))
        output = torch.sum(branch1*truck, dim=-1, keepdim=True) + self.p['bias']
     
        return output


    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params
