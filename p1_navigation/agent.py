import numpy as np

class Agent:

    def __init__(self):
        pass

    def select_action(self, state):
        return np.random.randint(4)

    def learn(self, state, action, reward, next_state):
        pass