###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb
# provided by udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from agent import Agent

class ContinuousControl:
    def __init__(self):
        self.env = UnityEnvironment(file_name='Reacher.app')
        self.brain_name = self.env.brain_names[0]
        # todo remove? self.brain = self.env.brains[self.brain_name]
        sample_batch_size = 64
        replay_buffer_size = 10e6
        gamma= 0.99
        tau= 0.001
        actor_learning_rate=10e-4
        critic_learning_rate=10e-3
        self.episodes = 2
        self.agent = Agent(sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate)
        self.scores = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train(self):
        for episode in range(self.episodes+1):
            state = self.env.reset(train_mode=True)[self.brain_name].vector_observations[0]
            score = 0
            print("train")

if __name__ == '__main__':
    ContinuousControl().train()