import numpy as np
import torch
import torch.nn as nn

# Calculate the range of values for uniform distributions
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

actor_net = {'fc1_units': 200, 'fc2_units': 150}
critic_net = {'fc1_units': 200, 'fc2_units': 150}

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, 
                 fc1_units = actor_net['fc1_units'], 
                 fc2_units = actor_net['fc2_units']):    
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Sequential(nn.Linear(state_size, fc1_units),
                                    nn.ReLU()) 
        
        self.layer2 = nn.Sequential(nn.Linear(fc1_units, fc2_units), 
                                    nn.ReLU())
        
        self.layer3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Apply to layers the specified weight initialization
        self.layer1[0].weight.data.uniform_(*hidden_init(self.layer1[0]))
        self.layer2[0].weight.data.uniform_(*hidden_init(self.layer2[0]))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""           
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.tanh(x)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed,
                 fc1_units=critic_net['fc1_units'],
                 fc2_units=critic_net['fc2_units']): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): seed
            num_agents (int): Total number of agents
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Sequential(nn.Linear(state_size * 2 + action_size * 2, fc1_units),
                                    nn.ReLU())
        
        self.layer2 = nn.Sequential(nn.Linear(fc1_units, fc2_units),
                                    nn.ReLU())
               
        self.layer3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()       
        
    def reset_parameters(self):
        # Apply to layers the specified weight initialization
        self.layer1[0].weight.data.uniform_(*hidden_init(self.layer1[0]))
        self.layer2[0].weight.data.uniform_(*hidden_init(self.layer2[0]))
        self.layer3.weight.data.uniform_(-3e-3, 3e-3)  
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-value."""
        xs = torch.cat((state, action), dim = 1)
        x = self.layer1(xs)
        x = self.layer2(x)
        output = self.layer3(x)
        return output