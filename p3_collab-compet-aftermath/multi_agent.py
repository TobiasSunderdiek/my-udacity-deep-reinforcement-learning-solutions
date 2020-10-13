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
        # In my first implementation I used a seed of 2, after having a look at a solution for this project
        # here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
        # I changed it to zero
        # Additionally, I had to add it here due to solution mentioned above, to use it within the replay buffer.
        seed = 0
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
        '''# I did not transform the observations from the buffer correct,
        # I got the correct transformation from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
        all_agents_states = np.concatenate(all_agents_states, axis=0)
        all_agents_actions = np.concatenate(all_agents_actions, axis=0)
        #all_agents_rewards = np.concatenate(all_agents_rewards, axis=0) # zero-dimensional arrays cannot be concatenated
        all_agents_next_states = np.concatenate(all_agents_next_states, axis=0)
        #all_agents_dones = np.concatenate(all_agents_dones, axis=0)'''
        
        self.replay_buffer.add(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)        

    def _sample_from_buffer(self):
        ''' neu
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
        '''
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

        ''' alt
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
        '''

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            # critic
            for actual_agent in range(self.num_agents):
                all_agents_states_buf, all_agents_actions, all_agents_rewards, all_agents_next_states_buf, all_agents_dones = self._sample_from_buffer()
                #all_agents_states 128,2,24
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

                '''
                # I did not transform the observations from the buffer correct,
                # I got the correct transformataion from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                
                    all_states_for_this_agent = torch.chunk(all_agents_states, 2, dim=1)
                    all_rewards_for_this_agent = all_agents_rewards[:, actual_agent].reshape(self.sample_batch_size, 1)
                    all_next_states_for_this_agent = torch.chunk(all_agents_next_states, 2, dim=1)
                    all_dones_for_this_agent = all_agents_dones[:, actual_agent].reshape(self.sample_batch_size, 1)
                '''

                # I got the implementation of updating the actor and critic from
                # the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
                # provided by udacity
                self.agents[actual_agent].critic_local.train()
                local_q_values = self.agents[actual_agent].critic_local(all_agents_states, all_agents_actions)

                '''
                # I called actor_target not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                this_agents_next_actions = [self.agents[actual_agent].actor_target(next_state_for_agent) for next_state_for_agent in all_next_states_for_this_agent]
                target_actions = torch.cat(this_agents_next_actions, dim=1).to(self.device)
                '''
                #todo
                # ich rufe den actor nicht im loop, sondern mit batch auf
                #todo hier fehlen die actions vom anderen agent, hier bekommt der critic
                # nur die Hälfte und daher exception
                # ich rufe daher wieder wir früher alle agents auf
                all_agents_next_actions = []
                for a in range(self.num_agents):
                    all_next_states_for_this_agent = torch.transpose(all_agents_next_states_buf, 0, 1)[a]
                    agent_target_action = self.agents[a].actor_target(all_next_states_for_this_agent)
                    all_agents_next_actions.append(agent_target_action)
                target_actions = torch.cat(all_agents_next_actions, 1) #128,2 und 128,2 (jeweils 1 Agent) -> 128,4
                
                target_q_values = all_rewards_for_this_agent + (self.gamma * self.agents[actual_agent].critic_target(all_agents_next_states, target_actions).detach() * (1 - all_dones_for_this_agent))
                critic_loss = F.mse_loss(local_q_values, target_q_values)

                self.agents[actual_agent].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 Continuous Control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(self.agents[actual_agent].critic_local.parameters(), 1)
                self.agents[actual_agent].critic_local_optimizer.step()
                self.agents[actual_agent].critic_local.eval()

                #actor
                # I called actor_local not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                
                #todo ich habe actor_local mit batch aufgerufen, mache ich jetzt auch wieder
                # aber auch wieder beide agents durchlaufen? weil geht ja in backprop
                #actions_local = [self.agents[actual_agent].actor_local(state_for_this_agent) for state_for_this_agent in all_states_for_this_agent]
                #actions_local_doubled = torch.cat(actions_local, 1)
                actions_local = []
                for b in range(self.num_agents):
                    #todo hier habe ich all-agents-states falsch behandelt, war aber nur in dieser Version
                    # da im original ich vorher nicht die variablen zugeweisen habe oben
                    # daher hier die original version aus dem buffer all_agents_states_buf
                    state_for_this_agent = torch.transpose(all_agents_states_buf, 0, 1)[b] # all_agents_states (ohne buf): 128,48 #batch 128, 48 ist obs
                    actions_local.append(self.agents[b].actor_local(state_for_this_agent)) # state_for_this_agent: 128
                
                #todo unnötige zuweisung
                actions_local_doubled = torch.cat(actions_local, 1) #128,2 und 128,2 (jeweils 1 Agent) -> 128,4


                actor_loss = -self.agents[actual_agent].critic_local(all_agents_states, actions_local_doubled).mean()
                self.agents[actual_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()
                self.agents[actual_agent].actor_local_optimizer.step()
                # After having a look in a solution for this project
                # here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                # I moved updating both target networks here. In my first implementation, I did this separately
                # at different places, the soft_update of the critic's target in this for-loop directly after the critic's part
                # and before the actor' part. And only the actor's part here after the actor.
                self.agents[actual_agent].soft_update(self.agents[actual_agent].critic_target, self.agents[actual_agent].critic_local, timestep)
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