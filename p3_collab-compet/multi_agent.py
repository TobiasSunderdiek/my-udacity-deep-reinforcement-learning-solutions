import torch
import torch.nn.functional as F
import numpy as np
from baselines.deepq.replay_buffer import ReplayBuffer

from agent import Agent

class MultiAgent:
    def __init__(self, observation_state_size, action_state_size, hyperparameter, num_agents, seed):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_agents = num_agents
        # I got using single param for all hyperparameters from udacity code review for my previous project
        self.agents = [Agent(observation_state_size, action_state_size, hyperparameter, seed) for i in range(num_agents)]
        self.sample_batch_size = hyperparameter['sample_batch_size']
        self.replay_buffer_size = hyperparameter['replay_buffer_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = hyperparameter['gamma']
        self.hyperparameter = hyperparameter

    def reset(self):
        for i in range(self.num_agents):
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agents[i].reset_noise

    def select_actions(self, all_agents_states, epsilon):
        return [self.agents[i].select_action(all_agents_states[i], epsilon) for i in range(self.num_agents)]

    def add_to_buffer(self, all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones):
        x = np.concatenate(all_agents_states, axis=0)
        next_x = np.concatenate(all_agents_next_states, axis=0)
        action = np.concatenate(all_agents_actions, axis=0)
        #achtung done, reward wird hier nicht concated, weil sonst error
        
        #e = self.experience(x, action, reward, next_x, done)
        self.replay_buffer.add(x, action, all_agents_rewards, next_x, all_agents_dones)        

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            # critic
            for num_agent in range(self.num_agents):
                x, actions, rewards, next_x, dones = self._sample_from_buffer()
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
                target_actions = torch.cat(next_actions, dim=1).to(self.device)  
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


                #my:
                
                #actor
                # Achtung jedne State einzeln: 
                actions_local = [self.agents[num_agent].actor_local(state) for state in states]
                actions_local = torch.cat(actions_local, 1)
                actor_loss = -self.agents[num_agent].critic_local(x, actions_local).mean()
                self.agents[num_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()
                self.agents[num_agent].actor_local_optimizer.step()
                self.agents[num_agent].soft_update(self.agents[num_agent].critic_target, self.agents[num_agent].critic_local, timestep)
                self.agents[num_agent].soft_update(self.agents[num_agent].actor_target, self.agents[num_agent].actor_local, timestep)
                '''
                self.agents[i].critic_local.train()
                #todo q value works with all obs, not single agent
                # + comment got it from lab maddpg implementation #todo
                #local_q_values = self.agents[i].critic_local(all_agents_states[i], all_agents_actions[i]) #todo
                local_q_values = self.agents[i].critic_local(torch.reshape(all_agents_states, (self.sample_batch_size, 48)), torch.reshape(all_agents_actions, (self.sample_batch_size, 4))) #todo/change insert full batch and reshape
                #next_actions = self.agents[i].actor_target(all_agents_next_states[i]) all_agents_next_actions
                all_agents_next_states_tranpose =  torch.transpose(all_agents_next_states, 0, 1)#[128, 2, 24] -> [2, 128, 24]
                all_agents_next_actions = []
                for j in range(self.num_agents):
                    next_states_agent_j = all_agents_next_states_tranpose[j]
                    agent_j_action = self.agents[j].actor_target(next_states_agent_j)
                    #print(f'agent_j_action {agent_j_action.shape}') #1024, 2
                    all_agents_next_actions.append(agent_j_action)
                all_agents_next_actions = torch.cat(all_agents_next_actions, 1) #das stimmt nicht?: 128,2 und 128,2 (jeweils 1 Agent) -> 128,4
                #print(f'all agents next actions resphaped for critic target {all_agents_next_actions.shape}')
                #all_agents_next_actions = self.agents[i].actor_target(all_agents_next_states) #todo for every agent i and sum?
                rewards_transpose = torch.transpose(all_agents_rewards, 0, 1) #(128,2,1) -> (2,128,1)
                reward_of_i = rewards_transpose[i]
                all_agents_dones_transpose = torch.transpose(all_agents_dones, 0, 1) #128,2,1 -> 2,128,1
                all_agents_dones_of_i = all_agents_dones_transpose[i]
                target_q_values = reward_of_i + (self.gamma * self.agents[i].critic_target(torch.reshape(all_agents_next_states, (self.sample_batch_size, 48)), all_agents_next_actions) * (1 - all_agents_dones_of_i)) #todo/change insert full batch, add all_agents_next_actions, add reward_of_i agent
                critic_loss = F.mse_loss(local_q_values, target_q_values)#todo huber loss
                self.agents[i].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 continuous control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(self.agents[i].critic_local.parameters(), 1)
                self.agents[i].critic_local_optimizer.step()
                self.agents[i].critic_local.eval()

                '''
           

    def _sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.hyperparameter['sample_batch_size'])

        # achtung umformung hierher verschoben funktioniert nicht
        # achtung dones extra nicht)

        x = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions_o = torch.from_numpy(np.vstack(actions)).float().to(self.device)
        rewards_o = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_x = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones_o = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        return x, actions_o, rewards_o, next_x, dones_o 

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