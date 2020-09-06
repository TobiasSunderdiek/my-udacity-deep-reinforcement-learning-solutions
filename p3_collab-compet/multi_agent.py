import torch
import torch.nn.functional as F
import numpy as np
from baselines.deepq.replay_buffer import ReplayBuffer

from agent import Agent

class MultiAgent:
    def __init__(self, observation_state_size, action_state_size, hyperparameter, num_agents):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_agents = num_agents
        # I got using single param for all hyperparameters from udacity code review for my previous project
        self.agents = [Agent(observation_state_size, action_state_size, hyperparameter) for i in range(num_agents)]
        self.sample_batch_size = hyperparameter['sample_batch_size']
        self.replay_buffer_size = hyperparameter['replay_buffer_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = hyperparameter['gamma']

    def reset_noise(self):
        for i in range(self.num_agents):
            print(i)
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agents[i].reset_noise

    def select_actions(self, all_agents_states, epsilon):
        return [self.agents[i].select_action(all_agents_states[i], epsilon) for i in range(self.num_agents)]

    def add_to_buffer(self, all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones):
        self.replay_buffer.add(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones = self._sample_from_buffer()
            # critic
            for i in range(self.num_agents):
                self.agents[i].critic_local.train()
                #todo q value works with all obs, not single agent
                # + comment got it from lab maddpg implementation #todo
                local_q_values = self.agents[i].critic_local(all_agents_states[i], all_agents_actions[i]) #todo
                next_actions = self.agents[i].actor_target(all_agents_next_states[i])
                target_q_values = all_agents_rewards[i] + (self.gamma * self.agents[i].critic_target(all_agents_next_states[i], next_actions) * (1 - all_agents_dones[i])) #todo
                critic_loss = F.mse_loss(local_q_values, target_q_values)#todo huber loss
                self.agents[i].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 continuous control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(self.agents[i].critic_local.parameters(), 1)
                self.agents[i].critic_local_optimizer.step()
                self.agents[i].critic_local.eval()
                self.agents[i].soft_update(self.agents[i].critic_target, self.agents[i].critic_local, timestep)

            # actor
            actions_local = self.agents[i].actor_local(all_agents_states[i])
            # todo critic works wiht all obs
            # + comment maddpg implementation todo
            actor_loss = -self.agents[i].critic_local(all_agents_states[i], actions_local).mean()
            self.agents[i].actor_local_optimizer.zero_grad()
            actor_loss.backward()
            self.agents[i].actor_local_optimizer.step()
            self.agents[i].soft_update(self.agents[i].actor_target, self.agents[i].actor_local, timestep)

    def _sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.sample_batch_size)
        # forgot to(device), after having a look at
        # https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L179
        # I added it here
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # dones: convert True/False to 0/1
        dones = torch.FloatTensor(np.where(dones == True, 1, 0)).unsqueeze(-1).to(self.device)
        return states, actions, rewards, next_states, dones