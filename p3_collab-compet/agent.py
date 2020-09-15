import torch
import torch.optim as optimizer
import numpy as np

from model import Actor, Critic
# Udacity Honor Code: Code copied
# I copied the class OUNoise from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/MADDPG.py#L179
import random
import copy
OU_THETA = 0.15         # how "strongly" the system reacts to perturbations
OU_SIGMA = 0.2
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()  
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state 

class Agent:
    def __init__(self, observation_state_size, action_space_size, hyperparameter, seed):#todo remove seed or add to doku
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L39
        # and https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L20
        # and https://pytorch.org/docs/stable/notes/cuda.html#best-practices
        # I added it here
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # forgot to use a seed, after having a look at: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
        # I added it here
        self.actor_local = Actor(observation_state_size, action_space_size, hyperparameter, seed).to(self.device)
        self.actor_target = Actor(observation_state_size, action_space_size, hyperparameter, seed).to(self.device)
        self.critic_local = Critic(observation_state_size*2, action_space_size*2, hyperparameter, seed).to(self.device)#todo *2 is num_agents, add dynamically
        self.critic_target = Critic(observation_state_size*2, action_space_size*2, hyperparameter, seed).to(self.device)#todo *2 is num_agents, add dynamically
        self.tau = hyperparameter['tau']
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), hyperparameter['actor_learning_rate'])
        # I got how to add weight decay like described in the paper
        # first from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L46
        # and second from here: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(),  hyperparameter['critic_learning_rate'])
        #todo add , weight_decay=0.0001) or remove comment above and add comment from solution
        # todo self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=0.2, theta=0.15)
        #self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=1.0, theta=0.15)
        #self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=hyperparameter['sigma'], theta=hyperparameter['theta'])
        # Udacity Honor Code: After having a look in a solution for this project
        # here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
        # I changed noise to class OUNoise
        self.noise = OUNoise(action_space_size, seed)
        self.update_every = hyperparameter['update_every']

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L64
    def select_action(self, state, epsilon):
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L66
        # I added it here
        state = torch.FloatTensor(state).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() # todo understand why is this directly the max action
        self.actor_local.train()
        #print(f'action vorher {action}')
        # Udacity Honor Code: After having a look in a solution for this project
        # here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
        # I removed decreasing noise by epsilon and used noise from OUNoise-Class from solution here
        tmp = self.noise.sample()# * epsilon
        action += tmp
        #action += self.noise() * epsilon #todo
        #print(f'action nachher {action} mit noise {tmp}')
        return np.clip(action, -1, 1)

    # I got the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L119
    def soft_update(self, target_network, local_network, timestep):
        # as mentioned in the udacity benchmark project of the previous project 2 continuous control, update weights only every x-timesteps
        if (timestep % self.update_every == 0):
            for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
                target_param.data.copy_((1-self.tau)*target_param.data + self.tau*local_param.data)
    '''def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)'''

    
    def reset_noise(self):
        self.noise.reset()