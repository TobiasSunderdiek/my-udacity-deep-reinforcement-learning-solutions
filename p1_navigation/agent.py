import numpy as np

from dqn import DQN

class Agent:

    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        self.action_size = output_size
        self.local_network = DQN(input_size, hidden_1_size, hidden_2_size, output_size)
        self.target_network = DQN(input_size, hidden_1_size, hidden_2_size, output_size)

    def select_action(self, state, epsilon):
        # Select action with epsilon-greedy-strategie:
        # First, create array with all possible actions
        # and their probability `epsilon/num_actions`
        # found the arange solution here: https://www.pluralsight.com/guides/different-ways-create-numpy-arrays
        action_pool = np.arange(self.action_size)
        action_probs = [epsilon/self.action_size for i in action_pool]
        # Second, get action with max Q-value from DQN
        # Probability for this action is `1-epsilon`
        max_q_value_action = np.argmax(self.local_network(state).data.cpu().numpy())
        max_q_value_action_prob = 1-epsilon
        action_pool = np.append(action_pool, max_q_value_action)
        action_probs = np.append(action_probs, max_q_value_action_prob)
        # Last, choose action according probability
        return np.random.choice(action_pool, p=action_probs)

    def learn(self, state, action, reward, next_state):
        pass

    def save_model(self):
        pass