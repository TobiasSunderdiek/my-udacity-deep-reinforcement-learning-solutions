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
                #local_q_values = self.agents[i].critic_local(all_agents_states[i], all_agents_actions[i]) #todo
                local_q_values = self.agents[i].critic_local(torch.reshape(all_agents_states, (128, 48)), torch.reshape(all_agents_actions, (128, 4))) #todo/change insert full batch and reshape
                #next_actions = self.agents[i].actor_target(all_agents_next_states[i]) all_agents_next_actions
                all_agents_next_states_tranpose =  torch.transpose(all_agents_next_states, 0, 1)#[128, 2, 24] -> [2, 128, 24]
                all_agents_next_actions = []
                for j in range(self.num_agents):
                    next_states_agent_j = all_agents_next_states_tranpose[j]
                    agent_j_action = self.agents[j].actor_target(next_states_agent_j)
                    all_agents_next_actions.append(agent_j_action)
                all_agents_next_actions = torch.cat(all_agents_next_actions, 1) #128,2 und 128,2 (jeweils 1 Agent) -> 128,4
                #all_agents_next_actions = self.agents[i].actor_target(all_agents_next_states) #todo for every agent i and sum?
                rewards_transpose = torch.transpose(all_agents_rewards, 0, 1) #(128,2,1) -> (2,128,1)
                reward_of_i = rewards_transpose[i]
                all_agents_dones_transpose = torch.transpose(all_agents_dones, 0, 1) #128,2,1 -> 2,128,1
                all_agents_dones_of_i = all_agents_dones_transpose[i]
                #target_q_values = all_agents_rewards[i] + (self.gamma * self.agents[i].critic_target(all_agents_next_states[i], next_actions) * (1 - all_agents_dones[i])) #todo
                target_q_values = reward_of_i + (self.gamma * self.agents[i].critic_target(torch.reshape(all_agents_next_states, (128,48)), all_agents_next_actions) * (1 - all_agents_dones_of_i)) #todo/change insert full batch, add all_agents_next_actions, add reward_of_i agent
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
            all_agent_states_transpose = torch.transpose(all_agents_states, 0, 1)
            all_actions_local = []
            for k in range(self.num_agents):
                actions_local = self.agents[k].actor_local(all_agent_states_transpose[k])
                all_actions_local.append(actions_local)
            all_actions_local = torch.cat(all_actions_local, 1) #128,2 und 128,2 (jeweils 1 Agent) -> 128,4
            # todo critic works wiht all obs
            # + comment maddpg implementation todo
            #actor_loss = -self.agents[i].critic_local(all_agents_states[i], actions_local).mean()
            actor_loss = -self.agents[i].critic_local(torch.reshape(all_agents_states, (128, 48)), all_actions_local).mean()
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