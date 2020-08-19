import torch.optim as optimizer
from baselines.deepq.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate):
        self.actor_local = ... #todo
        self.actor_target = ... #todo
        self.critic_local = ... #todo
        self.critic_target = ... #todo
        self.sample_batch_size = sample_batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), actor_learning_rate)
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(), critic_learning_rate)