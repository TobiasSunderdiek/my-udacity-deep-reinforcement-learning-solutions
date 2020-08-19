###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb
# provided by udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np

class ContinuousControl:
    def __init__(self):
        self.env = UnityEnvironment(file_name='Reacher.app')
        brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[brain_name]

    def train(self):
        print("train")

if __name__ == '__main__':
    ContinuousControl().train()