import torch.nn as nn
import torch
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
    def __init__(self, sensor_num1,sensor_num2,sensor_num3,sensor_num4, sensor_num5, 
                 sensor_num6, sensor_num7, sensor_num8, width=20, layers=5 ,actv=nn.Tanh()):
        super(IONet, self).__init__()
        self.actv=actv
        self.branch_net1 = FNN(ind=sensor_num1, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net2 = FNN(ind=sensor_num2, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net3 = FNN(ind=sensor_num3, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net4 = FNN(ind=sensor_num4, outd=width, layers=layers, width=width ,activation=actv)

        self.branch_net5 = FNN(ind=sensor_num5, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net6 = FNN(ind=sensor_num6, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net7 = FNN(ind=sensor_num7, outd=width, layers=layers, width=width ,activation=actv)
        self.branch_net8 = FNN(ind=sensor_num8, outd=width, layers=layers, width=width ,activation=actv)

        self.trunk_net1 = FNN(ind=1, outd=width, layers=layers, width=width  ,activation=actv)
        self.trunk_net2 = FNN(ind=1, outd=width, layers=layers, width=width  ,activation=actv)

        self.p = self.__init_params()
 
      
    def forward(self, X, label):
        rhs_l = X[0]
        branch1 =  self.branch_net1(rhs_l)
        rhs_r = X[1]
        branch2 = self.branch_net2(rhs_r)
        b_l = X[2]
        branch3 = self.branch_net3(b_l)
        b_r = X[3]
        branch4 = self.branch_net4(b_r)

        a_l = X[4]
        branch5 =  self.branch_net5(a_l)
        a_r = X[5]
        branch6 = self.branch_net6(a_r)
        boundary_l = X[6]
        branch7 = self.branch_net7(boundary_l)
        boundary_r = X[7]
        branch8 = self.branch_net8(boundary_r)
        
        x = X[8]
        if label == "l":  
            trunk = self.trunk_net1(x)
            output = torch.sum(branch1*branch2*branch3*branch4*branch5*branch6*branch7*branch8*trunk, dim=-1, keepdim=True) + self.p['bias1']
        elif label == 'r':
            trunk = self.trunk_net2(x)
            output = torch.sum(branch1*branch2*branch3*branch4*branch5*branch6*branch7*branch8*trunk, dim=-1, keepdim=True) + self.p['bias2']   

        return output


    def __init_params(self):
        params = nn.ParameterDict()
        params['bias1'] = nn.Parameter(torch.zeros([1]))
        params['bias2'] = nn.Parameter(torch.zeros([1]))
        return params




