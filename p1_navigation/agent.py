import numpy as np
import torch.optim as optimizer
import torch.nn.functional as F
from baselines.deepq.replay_buffer import ReplayBuffer

from dqn import DQN

class Agent:

    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size, buffer_size, sample_batch_size):
        self.action_size = output_size
        self.local_network = DQN(input_size, hidden_1_size, hidden_2_size, output_size)
        self.target_network = DQN(input_size, hidden_1_size, hidden_2_size, output_size)
        self.local_network.eval()
        self.optimizer = optimizer.Adam(self.local_network.parameters())
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.sample_batch_size = sample_batch_size

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

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self, update_target):
        self.local_network.train()
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.sample_batch_size)
        #loss = F.mse_loss(input, target)
        #loss.backward()
        self.optimizer.step()
        self.local_network.eval()
        if (update_target):
            # todo
            pass

    def save_model(self):
        pass