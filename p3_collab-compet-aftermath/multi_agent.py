###########################################
# This file is based on
# the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
# provided by Udacity
###########################################
import torch
import torch.nn.functional as F
import numpy as np
from agent import Agent
from baselines.deepq.replay_buffer import ReplayBuffer

class MultiAgent:
    def __init__(self, observation_state_size, action_state_size, hyperparameter, num_agents):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_agents = num_agents
        # I got using single param for all hyperparameters from udacity code review for my previous project: P2 Continuous Control
        self.agents = [Agent(observation_state_size, action_state_size, hyperparameter) for i in range(num_agents)]
        self.sample_batch_size = hyperparameter['sample_batch_size']
        self.replay_buffer_size = hyperparameter['replay_buffer_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = hyperparameter['gamma']
        self.hyperparameter = hyperparameter
        self.action_state_size = action_state_size

    def reset_noise(self):
        for i in range(self.num_agents):
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agents[i].reset_noise

    def select_actions(self, all_agents_states, epsilon):
        return [self.agents[i].select_action(all_agents_states[i], epsilon) for i in range(self.num_agents)]

    def add_to_buffer(self, all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones):
        self.replay_buffer.add(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)        

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

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            all_agents_states_buf, all_agents_actions, all_agents_rewards, all_agents_next_states_buf, all_agents_dones = self._sample_from_buffer()
            # critic
            for actual_agent in range(self.num_agents):
                all_agents_states = torch.reshape(all_agents_states_buf, (self.sample_batch_size, 48))
                all_agents_actions = torch.reshape(all_agents_actions, (self.sample_batch_size, 4))
                # todo achtung da ich unten im loop die next actions für alle agents hole
                # habe ich es hier wieder auskommentier
                # erheblichen einfluß, siehe kommentar for all_agents_next_states = ...
                #all_next_states_for_this_agent = torch.transpose(all_agents_next_states, 0, 1)[actual_agent]#[128, 2, 24] -> [2, 128, 24]
                all_rewards_for_this_agent = torch.transpose(all_agents_rewards, 0, 1)[actual_agent] #(128,2,1) -> (2,128,1)
                all_dones_for_this_agent = torch.transpose(all_agents_dones, 0, 1)[actual_agent] #128,2,1 -> 2,128,1
                #all_states_for_this_agent = torch.transpose(all_agents_states, 0, 1)[actual_agent]
                #todo achtung reihenfolge wichtige da all_next_states_for_this_agent auf altem Wert arbeitet
                # das könnte das Problem gewesen sein, da ich unten im loop all_next_states_for_this_agent
                # wieder verweende
                all_agents_next_states = torch.reshape(all_agents_next_states_buf, (self.sample_batch_size, 48))

                # I got the implementation of updating the actor and critic from
                # the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
                # provided by udacity
                self.agents[actual_agent].critic_local.train()
                local_q_values = self.agents[actual_agent].critic_local(all_agents_states, all_agents_actions)

                all_agents_next_actions = []
                for a in range(self.num_agents):
                    all_next_states_for_this_agent = torch.transpose(all_agents_next_states_buf, 0, 1)[a]
                    agent_target_action = self.agents[a].actor_target(all_next_states_for_this_agent)
                    all_agents_next_actions.append(agent_target_action)
                target_actions = torch.cat(all_agents_next_actions, 1)
                
                target_q_values = all_rewards_for_this_agent + (self.gamma * self.agents[actual_agent].critic_target(all_agents_next_states, target_actions).detach() * (1 - all_dones_for_this_agent))
                critic_loss = F.mse_loss(local_q_values, target_q_values)

                self.agents[actual_agent].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 Continuous Control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(self.agents[actual_agent].critic_local.parameters(), 1)
                self.agents[actual_agent].critic_local_optimizer.step()
                self.agents[actual_agent].critic_local.eval()
                self.agents[actual_agent].soft_update(self.agents[actual_agent].critic_target, self.agents[actual_agent].critic_local, timestep)

                #actor
                actions_local = []
                for b in range(self.num_agents):
                    state_for_this_agent = torch.transpose(all_agents_states_buf, 0, 1)[b]
                    actions_local.append(self.agents[b].actor_local(state_for_this_agent))
                local_actions = torch.cat(actions_local, 1)

                actor_loss = -self.agents[actual_agent].critic_local(all_agents_states, local_actions).mean()
                self.agents[actual_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()
                self.agents[actual_agent].actor_local_optimizer.step()
                self.agents[actual_agent].soft_update(self.agents[actual_agent].actor_target, self.agents[actual_agent].actor_local, timestep)
                
    def save(self):
        data = {}
        for i in range(self.num_agents):
            # 1. save weights and optimizer
            #    got this from Udacity's MADDPG-Lab-Implemetation
            # 2. merge dicts, got this from here: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
            data.update({'actor_local_'+str(i): self.agents[i].actor_local.state_dict(),
                        'critic_local_'+str(i): self.agents[i].critic_local.state_dict(),
                        'actor_local_optimizer_'+str(i): self.agents[i].actor_local_optimizer.state_dict(),
                        'critic_local_optimizer_'+str(i): self.agents[i].critic_local_optimizer.state_dict()
                        })
        torch.save(data, 'model.pth')