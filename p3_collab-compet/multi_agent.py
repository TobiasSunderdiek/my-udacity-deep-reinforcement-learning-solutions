import torch
import torch.nn.functional as F
import numpy as np
from baselines.deepq.replay_buffer import ReplayBuffer
from collections import namedtuple
import random

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
        self.tau = hyperparameter['tau']
        self.experience = namedtuple("Experience", field_names=["x", "action", "reward", "next_x", "done"])


    def reset_noise(self):
        for i in range(self.num_agents):
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agents[i].reset_noise

    def select_actions(self, all_agents_states, epsilon):
        return [self.agents[i].select_action(all_agents_states[i], epsilon) for i in range(self.num_agents)]

    def add_to_buffer(self, x, action, reward, next_x, done):
        x = np.concatenate(x, axis=0)
        next_x = np.concatenate(next_x, axis=0)
        action = np.concatenate(action, axis=0)
        
        self.replay_buffer.add(x, action, reward, next_x, done)

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            experiences = self._sample_from_buffer()
            # critic
            for num_agent in range(self.num_agents):
                x, actions, rewards, next_x, dones = experiences
   
                # Splits 'x' into a 'num_agents' of states
                states = torch.chunk(x, 2, dim = 1)
                # Splits 'next_x' into a 'num_agents' of next states
                next_states = torch.chunk(next_x, 2, dim = 1)
                
                # Get reward for each agent
                rewards = rewards[:,num_agent].reshape(rewards.shape[0],1)
                dones = dones[:,num_agent].reshape(dones.shape[0],1)
                
                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                next_actions = [self.agents[num_agent].actor_target(n_s) for n_s in next_states]
                target_actions = torch.cat(next_actions, dim=1)
                Q_targets_next = self.agents[num_agent].critic_target(next_x, target_actions)        
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))        
                # Compute critic loss
                Q_expected = self.agents[num_agent].critic_local(x, actions)
                critic_loss = F.mse_loss(Q_expected, Q_targets)        
                # Minimize the loss
                self.agents[num_agent].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                self.agents[num_agent].critic_local_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                # take the current states and predict actions
                actions_pred = [self.agents[num_agent].actor_local(s) for s in states]        
                actions_pred_ = torch.cat(actions_pred, dim=1)
                # -1 * (maximize) Q value for the current prediction
                actor_loss = -self.agents[num_agent].critic_local(x, actions_pred_).mean()        
                # Minimize the loss
                self.agents[num_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()        
                self.agents[num_agent].actor_local_optimizer.step()

                # ----------------------- update target networks ----------------------- #
                self.agents[num_agent].soft_update(self.agents[num_agent].critic_local, self.agents[num_agent].critic_target, self.tau)
                self.agents[num_agent].soft_update(self.agents[num_agent].actor_local, self.agents[num_agent].actor_target, self.tau)   

    def _sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.sample_batch_size)

        x = torch.from_numpy(np.vstack([state for state in states if state is not None])).float()
        actions_o = torch.from_numpy(np.vstack([e for e in actions if e is not None])).float()
        rewards_o = torch.from_numpy(np.vstack([e for e in rewards if e is not None])).float()
        next_x = torch.from_numpy(np.vstack([e for e in next_states if e is not None])).float()
        dones_o = torch.from_numpy(np.vstack([e for e in dones if e is not None]).astype(np.uint8)).float()
        return (x, actions_o, rewards_o, next_x, dones_o)

    def save(self):
        data = {}
        for i in range(self.num_agents):
            # 1. save weights and optimizer
            #    got this from Udacity's MADDPG-Lab-Implemetation
            # 2. merge dicts, got this from here: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
            data.update({'actor_local_'+i: self.agents[i].actor_local.state_dict(),
                        'critic_local_'+i: self.agents[i].critic_local.state_dict(),
                        'actor_local_optimizer_'+i: self.agents[i].actor_local_optimizer.state_dict(),
                        'critic_local_optimizer_'+i: self.agents[i].critic_local_optimizer.state_dict()
                        })
        torch.save(data, 'model.pth')