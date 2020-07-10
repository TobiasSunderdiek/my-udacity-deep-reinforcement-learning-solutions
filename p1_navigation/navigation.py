###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb
# provided by udacity
###########################################
import numpy as np
from unityagents import UnityEnvironment
from agent import Agent
from collections import deque
from epsilon import Epsilon

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

###Hyperparameter####
hidden_1_size = 37 * 2
hidden_2_size = 37 * 2
episodes = 2
max_timesteps_episode = 100
epsilon_start = 0.1
epsilon_max_decay_to = 0.01
#####################
input_size = 37
output_size = 4
agent = Agent(input_size, hidden_1_size, hidden_2_size, output_size)
epsilon = Epsilon(epsilon_start, epsilon_max_decay_to, max_timesteps_episode)
scores = deque(maxlen=100)
for episode in range(1, episodes+1):
    state = env.reset(train_mode=True)[brain_name].vector_observations[0]
    score = 0

    for timestep in range(0, max_timesteps_episode):
        current_epsilon = epsilon.calculate_for(timestep)
        action = agent.select_action(state, current_epsilon)
        env_info = env.step(action)[brain_name]
        next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
        agent.learn(state, action, reward, next_state)
        score += reward
        state = next_state
        if done:
            break

    scores.append(score)
    mean_score = np.mean(scores)
    if (episode % 10 == 0):
        print(f'Episode {episode} mean score {mean_score}', end='\r')
    if (len(scores) == 100 and mean_score == 13):
        print(f'Reached mean score of {mean_score} over last 100 episodes after episode {episode}')
        agent.save_model()
    
env.close()