import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)


class FNN(nn.Module):
    '''
        Fully connected neural networks.
    '''
    def __init__(self, ind, outd, layers=2, width=50, activation=nn.Tanh()):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
            
    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        return x

    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.activation
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i)] =  self.activation
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules



class IONet(nn.Module):
    def __init__(self, sensor_in,sensor_out, width, layers ,actv=nn.Tanh()):
        super(IONet, self).__init__()
        self.actv=actv
        self.branch_net1 = FNN(ind=sensor_in, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net2 = FNN(ind=sensor_out, outd=width, layers=layers, width=width ,activation=actv )

        self.trunk_net1 = FNN(ind=2, outd=width, layers=layers, width=width  ,activation=actv)
        self.trunk_net2 = FNN(ind=2, outd=width, layers=layers, width=width  ,activation=actv)

        self.p = self.__init_params()
 
      
    def forward(self, X, label):
        feature_up = X[0]
        branch1 =  self.branch_net1(feature_up)
        feature_down = X[1]
        branch2 = self.branch_net2(feature_down)
        
        x = X[2]
        if label == "inner":  
            trunk = self.trunk_net1(x)
            output = torch.sum(branch1*branch2*trunk, dim=-1, keepdim=True) + self.p['bias1']
        elif label == 'out':
            trunk = self.trunk_net2(x)
            output = torch.sum(branch1*branch2*trunk, dim=-1, keepdim=True) + self.p['bias2']   

        return output


    def __init_params(self):
        params = nn.ParameterDict()
        params['bias1'] = nn.Parameter(torch.zeros([1]))
        params['bias2'] = nn.Parameter(torch.zeros([1]))
        return params

