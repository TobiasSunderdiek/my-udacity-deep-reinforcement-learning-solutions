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

from MADDPG import MultiAgent
from epsilon import Epsilon
import torch #todo only for testing

class CollaborationAndCompetition:
    def __init__(self, hyperparameter):
        self.num_agents = 2
        observation_state_size = 24
        action_space_size = 2
        epsilon_start = 0.1
        epsilon_decay_rate = 0.995
        epsilon_max_decay_to = 0.01
        self.epsilon = Epsilon(epsilon_start, epsilon_decay_rate, epsilon_max_decay_to)
        self.episodes = 5_000
        seed = 0
        self.agents = MultiAgent(observation_state_size, action_space_size, hyperparameter, self.num_agents, seed)
        self.scores = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train(self, env):
        brain_name = env.brain_names[0]
        for episode in range(1, self.episodes+1):
            all_agents_states = env.reset(train_mode=True)[brain_name].vector_observations
            all_agents_score = np.zeros(self.num_agents)
            timestep = 0
            self.agents.reset()

            while True:
                current_epsilon = self.epsilon.calculate_for(timestep)
                all_agents_actions = self.agents.select_actions(all_agents_states, current_epsilon)
                #print(all_agents_actions)
                #all_agents_actions = np.asarray(all_agents_actions)
                #print("my")
                #print(all_agents_actions)
                #todo remove
                #if(count_rewards < 1024):
                #    all_agents_actions = np.random.randn(self.num_agents, 2) # select an action (for each agent)
                #    all_agents_actions = np.clip(all_agents_actions, -1, 1)
                #print('rand')
                #print(all_agents_actions)
                
                

                env_info = env.step(all_agents_actions)[brain_name]
                all_agents_next_states, all_agents_rewards, all_agents_dones = env_info.vector_observations, env_info.rewards, env_info.local_done
                
                self.agents.add_to_buffer(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)
                self.agents.learn(timestep)

                
                #neu self.agents.step(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)
                all_agents_score += all_agents_rewards
                #print(f'all_agents_rewards {all_agents_rewards}')
                #print(f'all_agents_score {all_agents_score}')
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
    #todo this are the params from the paper
    # https://arxiv.org/pdf/1706.02275.pdf
    hyperparameter = {'gamma': 0.99,
                      'sample_batch_size': 250,
                      # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                      # otherwise index error due to float
                      'replay_buffer_size': int(1e5),
                      'tau': 1e-3,
                      'actor_learning_rate': 1e-4,
                      'critic_learning_rate': 1e-3,
                      'update_every': 5,
                      'init_weights_variance': 3e-3,
                      'hidden_layer_1': 200,
                      'hidden_layer_2': 150,
                      'sigma': 0.2,
                      'theta': 0.15,
                    }

    env_filename = 'Tennis.app'
    env = UnityEnvironment(file_name=env_filename)
    CollaborationAndCompetition(hyperparameter).train(env)