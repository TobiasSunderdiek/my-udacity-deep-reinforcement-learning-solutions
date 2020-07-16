###########################################
# This file is based on the jupyter notebook
# https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb
# provided by udacity
###########################################
import numpy as np
from unityagents import UnityEnvironment
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from epsilon import Epsilon

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

###Hyperparameter####
hidden_1_size = 37 * 2
hidden_2_size = 37 * 2
epsilon_start = 0.1
epsilon_decay_rate = 0.995
epsilon_max_decay_to = 0.01
update_every = 4
buffer_size = 1_000_000
sample_batch_size = 64
gamma = 0.99
tau = 1e-3
learning_rate = 5e-4
#####################
episodes = 1_800
input_size = 37
output_size = 4
agent = Agent(input_size, hidden_1_size, hidden_2_size, output_size, buffer_size, sample_batch_size, gamma, tau, learning_rate)
epsilon = Epsilon(epsilon_start, epsilon_decay_rate, epsilon_max_decay_to)
scores = deque(maxlen=100)
writer = SummaryWriter()
for episode in range(1, episodes+1):
    state = env.reset(train_mode=True)[brain_name].vector_observations[0]
    score = 0
    losses = []
    rewards = []
    timestep = 0

    while True:
        current_epsilon = epsilon.calculate_for(timestep)
        action = agent.select_action(state, current_epsilon)
        env_info = env.step(action)[brain_name]
        next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
        agent.add_to_buffer(state, action, reward, next_state, done)
        update_target = timestep % update_every == 0
        loss = agent.learn(update_target)
        losses.append(loss.cpu().detach().numpy())
        rewards.append(reward)
        score += reward
        state = next_state
        timestep += 1
        if done:
            break

    scores.append(score)
    mean_score = np.mean(scores)
    if (episode % 10 == 0):
        print(f'Episode {episode} mean score {mean_score}', end='\r')
    if (len(scores) == 100 and mean_score >= 13):
        print(f'Reached mean score of {mean_score} over last 100 episodes after episode {episode}')
        agent.save_model()
        break
    writer.add_scalar("score", score, episode)

writer.close() 
env.close()