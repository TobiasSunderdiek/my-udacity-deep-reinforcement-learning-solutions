import torch
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
        weight_decay = 10e-2
        self.actor_local_optimizer = optimizer.Adam(self.actor_local.parameters(), actor_learning_rate)
        # I got how to add weight decay like described in the paper
        # first from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L46
        # and second from here: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
        self.critic_local_optimizer = optimizer.Adam(self.critic_local.parameters(), critic_learning_rate, weight_decay=weight_decay)

    def select_action(self, state):
        # select action probs with PPO
        # argmax auf die probs
        # damit dann dqn
        return 0.5 #todo implement

    def learn(self):
        pass #todo implement

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_model(self):
        #todo save only local weights?
        dict = {'actor_local': self.actor_local.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_local': self.critic_local.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_local_optimizer': self.actor_local_optimizer.state_dict(),
                'critic_local_optimizer': self.critic_local_optimizer.state_dict()}
        torch.save(dict, 'model.pth')