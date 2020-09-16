###########################################
# This file is based on
# the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
# provided by Udacity
###########################################
import torch
import torch.nn.functional as F
import numpy as np
from agent import Agent
from replay_buffer import ReplayBuffer

class MultiAgent:
    def __init__(self, observation_state_size, action_state_size, hyperparameter, num_agents):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_agents = num_agents
        # I got using single param for all hyperparameters from udacity code review for my previous project: P2 Continuous Control
        self.agents = [Agent(observation_state_size, action_state_size, hyperparameter) for i in range(num_agents)]
        self.sample_batch_size = hyperparameter['sample_batch_size']
        self.replay_buffer_size = hyperparameter['replay_buffer_size']
        # Udacity Honor Code: In my first implementation I used a seed of 2, after having a look at a solution for this project
        # here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
        # I changed it to zero
        # Also I had to add it here to due to solution mentioned above, to use it within the replay buffer.
        seed = 0
        self.replay_buffer = ReplayBuffer(action_state_size, self.replay_buffer_size, self.sample_batch_size, seed)
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
        all_agents_states = np.concatenate(all_agents_states, axis=0)
        all_agents_actions = np.concatenate(all_agents_actions, axis=0)
        all_agents_rewards = np.concatenate(all_agents_rewards, axis=0)
        all_agents_next_states = np.concatenate(all_agents_next_states, axis=0)
        all_agents_dones = np.concatenate(all_agents_dones, axis=0)
        
        self.replay_buffer.add(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)        

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) >= self.sample_batch_size):
            # critic
            for actual_agent in range(self.agents):
                (all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones) = self.replay_buffer.sample()
                
                



                # Udacity Honor Code: As mentioned in README, I had a bug in my implementation
                # add did not get a fresh sample for each agent, but used the same sample for both
                # Got the correct version from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                experiences = self.replay_buffer.sample()
                state, action, reward, next_state, done = experiences
                # Udacity Honor Code: As mentioned in README, I had a bug in the transformation of
                # the observations out of the replay buffer
                # I copied the correct transformation from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                x, actions, rewards, next_x, dones = state, action, reward, next_state, done
                states = torch.chunk(x, 2, dim = 1)
                next_states = torch.chunk(next_x, 2, dim = 1)
                rewards = rewards[:,num_agent].reshape(rewards.shape[0],1)
                dones = dones[:,num_agent].reshape(dones.shape[0],1)
                
                # I got the implementation of updating the actor and critic from
                # the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
                # provided by udacity
                self.agents[actual_agent].critic_local.train()
                local_q_values = self.agents[actual_agent].critic_local(all_agents_states, this_agents_doubled)

                # Udacity Honor Code: As mentioned in README, I called
                # actor_target not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                this_agents_next_actions = [self.agents[actual_agent].actor_target(next_state_for_agent) for next_state_for_agent in this_agents_next_states]
                target_actions = torch.cat(this_agents_next_actions, dim=1).to(self.device)  
                
                target_q_values = rewards + (self.gamma * self.agents[actual_agent].critic_target(all_agents_next_actions, target_actions).detach() * (1 - dones))
                critic_loss = F.mse_loss(local_q_values, target_q_values)

                self.agents['actual_agent'].critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 Continuous Control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(self.agents['actual_agent'].critic_local.parameters(), 1)
                self.agents['actual_agent'].critic_local_optimizer.step()
                self.agents['actual_agent'].critic_local.eval()

                #actor
                # Udacity Honor Code: As mentioned in README, I called
                # actor_local not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                actions_local = [self.agents['actual_agent'].actor_local(state_for_this_agent) for state_for_this_agent in this_agents_states]
                actions_local_doubled = torch.cat(actions_local, 1)

                actor_loss = -self.agents['actual_agent'].critic_local(all_agents_states, actions_local_doubled).mean()
                self.agents[actual_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()
                self.agents[actual_agent].actor_local_optimizer.step()
                # Udacity Honor Code: After having a look in a solution for this project
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