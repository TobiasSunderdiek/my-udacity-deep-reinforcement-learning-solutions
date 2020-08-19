import torch.optim as optimizer
from baselines.deepq.replay_buffer import ReplayBuffer

from model import Actor, Critic

class Agent:
    def __init__(self, observation_state_size, action_space_size, sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate):
        self.actor_local = Actor(observation_state_size, action_space_size)
        self.actor_target = Actor(observation_state_size, action_space_size)
        self.critic_local = Critic(observation_state_size, action_space_size)
        self.critic_target = Critic(observation_state_size, action_space_size)
        self.sample_batch_size = sample_batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), actor_learning_rate)
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(), critic_learning_rate)

    def select_action(self):
        return 0.5 #todo implement

    def save_model(self):
        pass #todo implement