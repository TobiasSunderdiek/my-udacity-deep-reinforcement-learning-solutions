###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb
# provided by udacity
###########################################
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]