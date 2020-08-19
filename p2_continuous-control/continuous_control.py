###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb
# provided by udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np

from agent import Agent

class ContinuousControl:
    def __init__(self):
        self.env = UnityEnvironment(file_name='Reacher.app')
        brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[brain_name]
        sample_batch_size = 64
        replay_buffer_size = 10e6
        gamma= 0.99
        tau= 0.001
        actor_learning_rate=10e-4
        critic_learning_rate=10e-3
        self.agent = Agent(sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate)

    def train(self):
        print("train")

if __name__ == '__main__':
    ContinuousControl().train()