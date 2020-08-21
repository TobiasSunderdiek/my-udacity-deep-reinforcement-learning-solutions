import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, output_size) # todo also add action?
        nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3) #todo correct this way?
        nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))

        return x

class Critic(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300+action_size, action_size) #todo action correct here?
        nn.init.uniform_(self.fc_3.weight, -3*10e-3, 3*10e-3) #todo correct this way?
        nn.init.uniform_(self.fc_3.bias, -3*10e-3, 3*10e-3)

    def forward(self, input, action):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = x + action
        x = self.fc_3(x) #todo leaky relu?

        return x

