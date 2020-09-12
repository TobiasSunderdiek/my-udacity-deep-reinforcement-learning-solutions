import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #todo
        #nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3)
        #nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)
        #todo
        # numpy.randn from jupyter make points
        # this is normal distribution, see: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html
        # with variance 1.0
        # so change to normal distribution
        nn.init.uniform_(self.fc_3.weight, -0.05, 0.05)
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
        #todo 
        #nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3)
        #nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)
        #nn.init.normal_(self.fc_3.weight, 0.0, init_weights_variance)
        nn.init.uniform_(self.fc_3.weight, -0.05, 0.05)

    def forward(self, input, action):
        x = F.leaky_relu(self.fc_1(input))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_3(x)

        return x