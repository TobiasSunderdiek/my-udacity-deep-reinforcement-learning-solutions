###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb
# provided by udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]