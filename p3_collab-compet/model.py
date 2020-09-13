import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Calculate the range of values for uniform distributions
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

actor_net = {'fc1_units': 200, 'fc2_units': 150}
critic_net = {'fc1_units': 200, 'fc2_units': 150}

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, action_size, seed, 
                 fc1_units = actor_net['fc1_units'], 
                 fc2_units = actor_net['fc2_units']):  
        super().__init__()
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size, 200)
        self.fc_2 = nn.Linear(200, 150)
        self.fc_3 = nn.Linear(150, action_size)
        #todo
        #nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3)
        #nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)
        #todo
        # numpy.randn from jupyter make points
        # this is normal distribution, see: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html
        # with variance 1.0
        # so change to normal distribution
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        nn.init.uniform_(self.fc_3.weight, -3e-3, 3e-3)
        #nn.init.normal_(self.fc_3.bias, 0.0, 0.8)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, action_size, seed,
                 fc1_units=critic_net['fc1_units'],
                 fc2_units=critic_net['fc2_units']): 
        super().__init__()
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size*2+action_size*2, 200)
        self.fc_2 = nn.Linear(200, 150)
        self.fc_3 = nn.Linear(150, 1) #todo understand why map to 1
        #todo 
        #nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3)
        #nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)
        #nn.init.normal_(self.fc_3.weight, 0.0, init_weights_variance)
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        nn.init.uniform_(self.fc_3.weight, -3e-3, 3e-3)

    def forward(self, input, action):
        x = torch.cat((input, action), dim=1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x