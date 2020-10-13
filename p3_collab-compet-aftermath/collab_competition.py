###########################################
# This file is based on
# - the jupyter notebook https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/Tennis.ipynb
# - the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
# both provided by Udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from multi_agent import MultiAgent
from epsilon import Epsilon

class CollaborationAndCompetition:
    def __init__(self, hyperparameter):
        self.num_agents = 2
        observation_state_size = 24
        action_space_size = 2
        epsilon_start = 0.1
        epsilon_decay_rate = 0.995
        epsilon_max_decay_to = 0.01
        self.epsilon = Epsilon(epsilon_start, epsilon_decay_rate, epsilon_max_decay_to)
        # The provided charts from the Udacity Benchmark Implementation for this project provided in the course
        # `Benchmark Implementation` show, that solving this problem < 5.000 episodes is possible. During my training
        # I was not sure if my implementation needs more episodes, therfore increased it.
        self.episodes = 10_000
        self.agents = MultiAgent(observation_state_size, action_space_size, hyperparameter, self.num_agents)
        self.scores = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train(self, env):
        brain_name = env.brain_names[0]
        for episode in range(1, self.episodes+1):
            all_agents_states = env.reset(train_mode=True)[brain_name].vector_observations
            all_agents_score = np.zeros(self.num_agents)
            timestep = 0
            self.agents.reset_noise()

            while True:
                current_epsilon = self.epsilon.calculate_for(timestep)
                all_agents_actions = self.agents.select_actions(all_agents_states, current_epsilon)
                all_agents_actions = np.asarray(all_agents_actions)
                env_info = env.step(all_agents_actions)[brain_name]
                all_agents_next_states, all_agents_rewards, all_agents_dones = env_info.vector_observations, env_info.rewards, env_info.local_done
                self.agents.add_to_buffer(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)
                self.agents.learn(timestep)
                all_agents_score += all_agents_rewards
                all_agents_states = all_agents_next_states
                timestep += 1
                if any(all_agents_dones):
                    break
            max_score = max(all_agents_score)
            self.scores.append(max_score)
            mean_score = np.mean(self.scores)
            if (episode % 100 == 0):
                print(f'Episode {episode} mean score {mean_score}')
            if (len(self.scores) == 100 and mean_score >= 0.5):
                print(f'Reached mean score of {mean_score} over last 100 episodes after episode {episode}')
                self.agents.save()
                break
            for i in range(self.num_agents):
                self.writer.add_scalar(f'score_agent_{i}', all_agents_score[i], episode)
            self.writer.add_scalar("mean_score", mean_score, episode)

        self.writer.close()

        return max_score

if __name__ == '__main__':
    hyperparameter = {'gamma': 0.99,
                      'sample_batch_size': 128,
                      # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                      # otherwise index error due to float
                      'replay_buffer_size': int(1e6),
                      'tau': 0.01,
                      'actor_learning_rate': 0.0001,
                      'critic_learning_rate': 0.0003,
                      'update_every': 1,
                      'init_weights_variance': 3*10e-3,
                      'hidden_layer_1': 400,
                      'hidden_layer_2': 300,
                      'sigma': 0.2,
                      'theta': 0.15,
                    }

    env_filename = 'Tennis.app'
    env = UnityEnvironment(file_name=env_filename)
    CollaborationAndCompetition(hyperparameter).train(env)