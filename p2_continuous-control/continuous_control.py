###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb
# provided by udacity
###########################################

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from epsilon import Epsilon

class ContinuousControl:
    def __init__(self, hyperparameter):
        observation_state_size = 33
        action_space_size = 4
        epsilon_start = 0.1
        epsilon_decay_rate = 0.995
        epsilon_max_decay_to = 0.01
        self.epsilon = Epsilon(epsilon_start, epsilon_decay_rate, epsilon_max_decay_to)
        # after reading the advice from the Udacity Knowledge Base https://knowledge.udacity.com/questions/277763
        # for this project, that training is possible <500 episodes, I choosed this value
        self.episodes = 500
        # Changes from udacity code review: In my first implementation, I passed all hyperparameters as single parameters to the agent
        # udacity code review suggests me to remove this and use one single param which has all hyperparameters in a map instead
        self.agent = Agent(observation_state_size, action_space_size, hyperparameter)
        self.scores = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train(self, env):
        brain_name = env.brain_names[0]
        for episode in range(1, self.episodes+1):
            state = env.reset(train_mode=True)[brain_name].vector_observations[0]
            score = 0
            timestep = 0
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agent.reset_noise

            while True:
                current_epsilon = self.epsilon.calculate_for(timestep)
                action = self.agent.select_action(state, current_epsilon)
                env_info = env.step(action)[brain_name]
                next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
                self.agent.add_to_buffer(state, action, reward, next_state, done)
                self.agent.learn(timestep)
                score += reward
                state = next_state
                timestep += 1
                if done:
                    break

            self.scores.append(score)
            mean_score = np.mean(self.scores)
            if (episode % 10 == 0):
                print(f'Episode {episode} mean score {mean_score}')
            if (len(self.scores) == 100 and mean_score >= 30):
                print(f'Reached mean score of {mean_score} over last 100 episodes after episode {episode}')
                self.agent.save_model()
                break
            self.writer.add_scalar("score", score, episode)
            self.writer.add_scalar("replay_buffer_fill_level", len(self.agent.replay_buffer), episode)

        self.writer.close()

        return score

if __name__ == '__main__':
    hyperparameter = {'gamma': 0.99,
                      'sample_batch_size': 128,
                      # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                      # otherwise index error due to float
                      'replay_buffer_size': int(1e6),
                      'tau': 0.01,
                      'actor_learning_rate': 0.0001,
                      'critic_learning_rate': 0.0003,
                      'update_every': 10
                    }
    env_filename = 'Reacher_Linux_NoVis/Reacher.x86_64'
    env = UnityEnvironment(file_name=env_filename)
    ContinuousControl(hyperparameter).train(env)