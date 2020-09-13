import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
        and from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L12
    """
    def __init__(self, input_size, action_size, init_weights_variance, hidden_layer_1, hidden_layer_2, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L26
        # and added it as I did not use a seed in my implementation
        #todo # Changes from udacity code review: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size, hidden_layer_1)
        self.fc_2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc_3 = nn.Linear(hidden_layer_2, action_size)
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

class Critic(nn.Module):
    """ I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
        and copied it from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L44
    """
    def __init__(self, input_size, action_size, init_weights_variance, hidden_layer_1, hidden_layer_2, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L58
        # and added it as I did not use a seed in my implementation
        #todo # Changes from udacity code review: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size, hidden_layer_1)
        self.fc_2 = nn.Linear(hidden_layer_1+action_size, hidden_layer_2)
        self.fc_3 = nn.Linear(hidden_layer_2, 1) #todo understand why map to 1
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, input, action):
        x = F.leaky_relu(self.fc_1(input))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_3(x)

        return x