import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.1, alpha=0.05, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self._epsilon_greedy_policy(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.Q[state][action] + self.alpha * (reward + self.gamma * self._expected_value_sum(next_state) - self.Q[state][action])

    def _epsilon_greedy_policy(self, state):
        return np.random.choice(self.nA)

    def _expected_value_sum(self, next_state):
        sum = 0
        for action in range(self.nA):
            sum += self._probability_of_action_in_next_step(next_state, action) * self.Q[next_state][action]
        return sum

    def _probability_of_action_in_next_step(self, next_state, action):
        if (self.Q[next_state][action] == max(self.Q[next_state]):
            probability = 1 - self.epsilon + (self.epsilon / self.nA)
        else:
            probability = self.epsilon / self.nA

        return probability