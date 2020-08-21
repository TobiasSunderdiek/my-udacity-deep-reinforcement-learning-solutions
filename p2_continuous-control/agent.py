import torch
import torch.optim as optimizer
import numpy as np
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from model import Actor, Critic

class Agent:
    def __init__(self, observation_state_size, action_space_size, sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate):
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L39
        # and https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L20
        # I added it here
        self.device = torch.cuda() if torch.cuda.is_available() else torch.device('cpu') #todo more to(device)?
        self.actor_local = Actor(observation_state_size, action_space_size).to(self.device)
        self.actor_target = Actor(observation_state_size, action_space_size).to(self.device)
        self.critic_local = Critic(observation_state_size, action_space_size).to(self.device)
        self.critic_target = Critic(observation_state_size, action_space_size).to(self.device)
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
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size), sigma=0.2, theta=0.15) #todo mean correct?

    def select_action(self, state):
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L66
        # I added it here
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor_local(state).data.cpu().numpy() #todo that's directly the max action?
        return action + self.noise()

    def learn(self):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.replay_buffer_size):
            self.actor_local_optimizer.zero_grad()
            self.critic_local_optimizer.zero_grad()
            states, actions, rewards, next_states, dones = self._sample_from_buffer()

            self.actor_local_optimizer.step()
            self.critic_local_optimizer.step()
            #todo implement

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_model(self):
        #todo save only local weights?
        dict = {'actor_local': self.actor_local.state_dict().cpu(),
                'actor_target': self.actor_target.state_dict().cpu(),
                'critic_local': self.critic_local.state_dict().cpu(),
                'critic_target': self.critic_target.state_dict().cpu(),
                'actor_local_optimizer': self.actor_local_optimizer.state_dict(),
                'critic_local_optimizer': self.critic_local_optimizer.state_dict()}
        torch.save(dict, 'model.pth')

    def _sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.sample_batch_size)
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L179
        # I added it here
        states = torch.FloatTensor(states).to(self.device)
        # found the unsqueeze and gather solution here: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        actions = torch.LongTensor(actions).unsqueeze(-1) #todo hier auch unsqueeze nötig?
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        # dones: convert True/False to 0/1
        dones = torch.FloatTensor(np.where(dones == True, 1, 0))
        return states, actions, rewards, next_states, dones