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

class ContinuousControl:
    def __init__(self, hyperparameter):
        #self.env = UnityEnvironment(file_name='../../../Reacher.app')
        #self.env = UnityEnvironment(file_name='Reacher.app')
        #self.env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
        self.env = UnityEnvironment(file_name='../../../Reacher_Linux_NoVis/Reacher.x86_64')
        self.brain_name = self.env.brain_names[0]
        observation_state_size = 33
        action_space_size = 4
        gamma= hyperparameter['gamma']
        sample_batch_size = hyperparameter['sample_batch_size']
        replay_buffer_size = hyperparameter['replay_buffer_size']
        tau = hyperparameter['tau']
        actor_learning_rate = hyperparameter['actor_learning_rate']
        critic_learning_rate = hyperparameter['critic_learning_rate']
        update_every = hyperparameter['update_every']
        self.episodes = 2_000#4_000 #todo
        self.agent = Agent(observation_state_size, action_space_size, sample_batch_size, replay_buffer_size, gamma, tau, actor_learning_rate, critic_learning_rate, update_every)
        self.scores = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train(self):
        for episode in range(1, self.episodes+1):
            state = self.env.reset(train_mode=True)[self.brain_name].vector_observations[0]
            score = 0
            timestep = 0
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agent.reset_noise

            while True:
                action = self.agent.select_action(state)
                env_info = self.env.step(action)[self.brain_name]
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
            #self.writer.add_scalar("score", score, episode)
            #self.writer.add_scalar("replay_buffer_fill_level", len(self.agent.replay_buffer), episode)

        self.writer.close()
        self.env.close()
        return score

if __name__ == '__main__':
    hyperparameter = {'gamma': 0.99,
                      'sample_batch_size': 64,
                      # cast buffer size to int, I got the casting from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L12
                      # otherwise index error due to float
                      'replay_buffer_size': int(1e5),
                      'tau': 0.001,
                      'actor_learning_rate': 10e-4,
                      'critic_learning_rate': 10e-3,
                      'update_every': 10
                    }
    ContinuousControl(hyperparameter).train()