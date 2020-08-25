import torch
import torch.nn as nn
import torch.nn.functional as F

# I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
# and from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L12
class Actor(nn.Module):
    def __init__(self, input_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed) # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L26
        self.fc_1 = nn.Linear(input_size, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, action_size)
        nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3) #todo correct this way?
        nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

# I got this from the description in the paper: https://arxiv.org/pdf/1509.02971.pdf
# and copied it from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L44
class Critic(nn.Module):
    def __init__(self, input_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed) # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py#L58
        self.fc_1 = nn.Linear(input_size, 400)
        self.fc_2 = nn.Linear(400+action_size, 300)
        self.fc_3 = nn.Linear(300, 1) #todo why map to 1?
        nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3) #todo correct this way?
        nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)

    def forward(self, input, action):
        x = F.leaky_relu(self.fc_1(input))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc_2(x))
        x = x + action
        x = self.fc_3(x) # why no activation?

        return x

