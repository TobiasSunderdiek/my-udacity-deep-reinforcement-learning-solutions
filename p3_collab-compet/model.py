import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Udacity Honor Code: I did not initialize layer 1 and layer 2 within my first implementation,
# after having a look at the solution for this project from
# here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I added it
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    # I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
    # and from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L12
    def __init__(self, input_size, action_size, hyperparameter, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L26
        # and added it as I did not use a seed in my implementation
        # Changes from udacity code review for my previous project: P2 Continuous Control: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size, hyperparameter['hidden_layer_1'])
        self.fc_2 = nn.Linear(hyperparameter['hidden_layer_1'], hyperparameter['hidden_layer_2'])
        self.fc_3 = nn.Linear(hyperparameter['hidden_layer_2'], action_size)
        # Udacity Honor Code: I did not initialize layer 1 and layer 2 within my first implementation,
        # after having a look at the solution for this project from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I added it
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        nn.init.uniform_(self.fc_3.weight, -hyperparameter['init_weights_variance'], hyperparameter['init_weights_variance'])

    def forward(self, input):
        # Udacity Honor Code: I did not use relu as activation function, after having a look at the solution for this project from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I changed
        # the following both activation functions to relu
        x = F.relu(self.fc_1(input))
        x = F.relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

class Critic(nn.Module):
    # I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
    # and copied it from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L44
    def __init__(self, input_size, action_size, hyperparameter, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L58
        # and added it as I did not use a seed in my implementation
        # Changes from udacity code review for my previous project: P2 Continuous Control: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        # Udacity Honor Code: I added `action_size` to the first layer of my implementation, after having a look at the solution for this project from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I added
        # it to the input
        self.fc_1 = nn.Linear(input_size+action_size, hyperparameter['hidden_layer_1'])
        self.fc_2 = nn.Linear(hyperparameter['hidden_layer_1'], hyperparameter['hidden_layer_2'])
        self.fc_3 = nn.Linear(hyperparameter['hidden_layer_2'], 1)
        # Udacity Honor Code: I did not initialize layer 1 and layer 2 within my first implementation,
        # after having a look at the solution from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I added it
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        nn.init.uniform_(self.fc_3.weight, -hyperparameter['init_weights_variance'], hyperparameter['init_weights_variance'])

    def forward(self, input, action):
        # Udacity Honor Code: I added `action_size` to the first layer of my implementation, after having a look at the solution for this project from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I added
        # it to the input
        x = torch.cat((input, action), dim=1)
        # Udacity Honor Code: I did not use relu as activation function, after having a look at the solution for this project from
        # here https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition, I changed
        # the following both activation functions to relu
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x
'''
class Actor(nn.Module):
    """ I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
        and from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L12
    """
    def __init__(self, input_size, action_size, hyperparameter, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L26
        # and added it as I did not use a seed in my implementation
        #todo # Changes from udacity code review: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size, hyperparameter['hidden_layer_1'])
        self.fc_2 = nn.Linear(hyperparameter['hidden_layer_1'], hyperparameter['hidden_layer_2'])
        self.fc_3 = nn.Linear(hyperparameter['hidden_layer_2'], action_size)
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
        nn.init.uniform_(self.fc_3.weight, -hyperparameter['init_weights_variance'], hyperparameter['init_weights_variance'])
        #nn.init.normal_(self.fc_3.bias, 0.0, 0.8)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

class Critic(nn.Module):
    """ I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
        and copied it from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L44
    """
    def __init__(self, input_size, action_size, hyperparameter, seed):
        super().__init__()
        # I got using a seed from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L58
        # and added it as I did not use a seed in my implementation
        #todo # Changes from udacity code review: In my first implementation I assigned the seed to a class variable(but never used it) and the
        # udacity code review suggests me to remove this class variable as I do not need it in my implementation
        torch.manual_seed(seed)
        self.fc_1 = nn.Linear(input_size+action_size, hyperparameter['hidden_layer_1'])
        self.fc_2 = nn.Linear(hyperparameter['hidden_layer_1'], hyperparameter['hidden_layer_2'])
        self.fc_3 = nn.Linear(hyperparameter['hidden_layer_2'], 1) #todo understand why map to 1
        #todo 
        #nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3)
        #nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)
        #nn.init.normal_(self.fc_3.weight, 0.0, init_weights_variance)
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        nn.init.uniform_(self.fc_3.weight, -hyperparameter['init_weights_variance'], hyperparameter['init_weights_variance'])

    def forward(self, input, action):
        x = torch.cat((input, action), dim=1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x
'''