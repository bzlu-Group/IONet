import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

class ION_Stacked(nn.Module):
    def __init__(self, sensor_in,sensor_out, m=20,actv=nn.Tanh()):
        super(ION_Stacked, self).__init__()
        self.actv=actv
        self.b1linear_input = nn.Linear(sensor_in,m)    
        self.b1linear1 =nn.Linear(m,m)
        self.b1linear2 = nn.Linear(m,m)
        self.b1linear3 =nn.Linear(m,m)
        self.b1linear4 = nn.Linear(m,m)

        self.b2linear_input = nn.Linear(sensor_out,m)    
        self.b2linear1 =nn.Linear(m,m)
        self.b2linear2 = nn.Linear(m,m)
        self.b2linear3 =nn.Linear(m,m)
        self.b2linear4 = nn.Linear(m,m)

        self.t1linear_input = nn.Linear(6,m)    
        self.t1linear1 =nn.Linear(m,m)
        self.t1linear2 = nn.Linear(m,m)
        self.t1linear3 =nn.Linear(m,m)
        self.t1linear4 = nn.Linear(m,m)

        self.t2linear_input = nn.Linear(6,m)    
        self.t2linear1 =nn.Linear(m,m)
        self.t2linear2 = nn.Linear(m,m)
        self.t2linear3 =nn.Linear(m,m)
        self.t2linear4 = nn.Linear(m,m)

        self.p = self.__init_params()
 
      
    def forward(self, X, label):
        feature_up = X[0]
        y = self.b1linear1(self.actv(self.b1linear_input(feature_up)))
        y = self.b1linear2(self.actv(y))
        y = self.b1linear3(self.actv(y))
        branch1 =  self.b1linear4(self.actv(y))
        
        feature_down = X[1]
        y = self.b2linear1(self.actv(self.b2linear_input(feature_down)))
        y = self.b2linear2(self.actv(y))
        y = self.b2linear3(self.actv(y))
        branch2 = self.b2linear4(self.actv(y))
        
        x = X[2]
        if label == "inner":  
            y = self.t1linear1(self.actv(self.t1linear_input(x)))
            y = self.t1linear2(self.actv(y))
            y = self.t1linear3(self.actv(y))
            truck = self.t1linear4(self.actv(y))
            output = torch.sum(branch1*branch2*truck, dim=-1, keepdim=True) + self.p['bias1']
        elif label == 'out':
            y = self.t2linear1(self.actv(self.t2linear_input(x)))
            y = self.t2linear2(self.actv(y))
            y= self.t2linear3(self.actv(y))
            truck = self.t2linear4(self.actv(y))
            output = torch.sum(branch1*branch2*truck, dim=-1, keepdim=True) + self.p['bias2']   

        return output


    def __init_params(self):
        params = nn.ParameterDict()
        params['bias1'] = nn.Parameter(torch.zeros([1]))
        params['bias2'] = nn.Parameter(torch.zeros([1]))
        return params

