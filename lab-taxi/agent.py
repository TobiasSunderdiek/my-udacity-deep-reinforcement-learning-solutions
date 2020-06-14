import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.0105, epsilon_decay_to=0.0100, epsilon_decay_rate=0.000001, alpha=0.0320, alpha_decay_to=0.0310, alpha_decay_rate=0.000025, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA, dtype= np.float128)) # float128 due to RuntimeWarning: overflow encountered in double_scalars in Q-Table update
        self.epsilon = epsilon
        self.epsilon_decay_to = epsilon_decay_to
        self.epsilon_decay_rate = (epsilon-epsilon_decay_to)/20000#epsilon_decay_rate
        self.alpha = alpha
        self.alpha_decay_to = alpha_decay_to
        self.alpha_decay_rate = (alpha-alpha_decay_to)/20000#alpha_decay_rate
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

        if (done):
            self._decrease_epsilon(reward);
            self._decrease_alpha();

    def _epsilon_greedy_policy(self, state):
        actions = [np.argmax(self.Q[state]), 0, 1, 2, 3, 4, 5]
        probabilities = [1-self.epsilon, self.epsilon/self.nA, self.epsilon/self.nA, self.epsilon/self.nA, self.epsilon/self.nA, self.epsilon/self.nA, self.epsilon/self.nA]
        return np.random.choice(actions, p=probabilities)

    def _expected_value_sum(self, next_state):
        sum = 0
        for action in range(self.nA):
            sum += self._probability_of_action_in_next_step(next_state, action) * self.Q[next_state][action]
        return sum

    def _probability_of_action_in_next_step(self, next_state, action):
        if (self.Q[next_state][action] == max(self.Q[next_state])):
            probability = 1 - self.epsilon + (self.epsilon / self.nA)
        else:
            probability = self.epsilon / self.nA
        return probability

    def _decrease_epsilon(self, reward):
        if (self.epsilon > self.epsilon_decay_to):
            self.epsilon -= self.epsilon_decay_rate

    def _decrease_alpha(self):
        if (self.alpha > self.alpha_decay_to):
            self.alpha -= self.alpha_decay_rate
