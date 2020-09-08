import torch
import torch.optim as optimizer
import numpy as np
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from model import Actor, Critic

class Agent:
    def __init__(self, observation_state_size, action_space_size, hyperparameter):
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L39
        # and https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L20
        # and https://pytorch.org/docs/stable/notes/cuda.html#best-practices
        # I added it here
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # forgot to use a seed, after having a look at: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
        # I added it here
        seed = 2
        self.actor_local = Actor(observation_state_size, action_space_size, seed).to(self.device)
        self.actor_target = Actor(observation_state_size, action_space_size, seed).to(self.device)
        self.critic_local = Critic(observation_state_size*2, action_space_size*2, seed).to(self.device)#todo *2 is num_agents, add dynamically
        self.critic_target = Critic(observation_state_size*2, action_space_size*2, seed).to(self.device)#todo *2 is num_agents, add dynamically
        self.tau = hyperparameter['tau']
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), hyperparameter['actor_learning_rate'])
        # I got how to add weight decay like described in the paper
        # first from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L46
        # and second from here: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(),  hyperparameter['critic_learning_rate'], weight_decay=0.0001)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=0.2, theta=0.15)
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
        action += self.noise() * epsilon
        return np.clip(action, -1, 1)

    # I got the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L119
    def soft_update(self, target_network, local_network, timestep):
        # as mentioned in the udacity benchmark project of the previous project 2 continuous control, update weights only every x-timesteps
        if (timestep % self.update_every == 0):
            for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
                target_param.data.copy_((1-self.tau)*target_param.data + self.tau*local_param.data)

    def save_model(self):
        # save only local weights
        # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
        dict = {'actor_local': self.actor_local.state_dict(),
                'critic_local': self.critic_local.state_dict()}
        torch.save(dict, 'model.pth')

    def reset_noise(self):
        self.noise.reset()