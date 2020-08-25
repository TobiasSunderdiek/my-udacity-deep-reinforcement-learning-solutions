import torch
import torch.optim as optimizer
import torch.nn.functional as F
import numpy as np
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from model import Actor, Critic

class Agent:
    def __init__(self, observation_state_size, action_space_size, sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate):
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
        self.critic_local = Critic(observation_state_size, action_space_size, seed).to(self.device)
        self.critic_target = Critic(observation_state_size, action_space_size, seed).to(self.device)
        self.sample_batch_size = sample_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), actor_learning_rate)
        # I got how to add weight decay like described in the paper
        # first from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L46
        # and second from here: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(), critic_learning_rate, weight_decay=10e-2)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=0.2, theta=0.15) # todo values centered around 0 like in the paper?

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L64
    def select_action(self, state):
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L66
        # I added it here
        state = torch.FloatTensor(state).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() # todo why is this directly the max action
        self.actor_local.train()
        action += self.noise()
        return np.clip(action, -1, 1)

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    def learn(self):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.replay_buffer_size):
            # critic
            self.critic_local.train()
            states, actions, rewards, next_states, dones = self._sample_from_buffer()
            local_q_values = self.critic_local(states, actions)
            next_actions = self.actor_target(next_states)
            target_q_values = rewards + (self.gamma * self.critic_target(next_states, next_actions) * (1 - dones))
            critic_loss = F.mse_loss(local_q_values, target_q_values)
            self.critic_local_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_local_optimizer.step()
            self.critic_local.eval()
            self.soft_update(self.critic_target, self.critic_local)

            # actor
            actions_local = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_local).mean()
            self.actor_local_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_local_optimizer.step()
            self.soft_update(self.actor_target, self.actor_local)

    # I got the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L119
    def soft_update(self, target_network, local_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_((1-self.tau)*target_param.data + self.tau*local_param.data)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_model(self):
        # save only local weights
        # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
        dict = {'actor_local': self.actor_local.state_dict().cpu(),
                'critic_local': self.critic_local.state_dict().cpu()}
        torch.save(dict, 'model.pth')

    def _sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.sample_batch_size)
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L179
        # I added it here
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # dones: convert True/False to 0/1
        dones = torch.FloatTensor(np.where(dones == True, 1, 0)).unsqueeze(-1).to(self.device)
        return states, actions, rewards, next_states, dones

    def reset_noise(self):
        self.noise.reset()