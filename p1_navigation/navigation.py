###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb
# provided by udacity
###########################################
import numpy as np
from unityagents import UnityEnvironment
from agent import Agent
from collections import deque

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent()
episodes = 2
max_timesteps_episode = 100
scores = deque(maxlen=100)
for episode in range(1, episodes+1):
    state = env.reset(train_mode=True)[brain_name].vector_observations[0]
    score = 0

    for timestep in range(0, max_timesteps_episode):
        action = agent.select_action(state)
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
        # TODO save model weights
    
env.close()