###########################################
# This file is based on
# the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
# provided by Udacity
###########################################

import torch
import torch.nn.functional as F
import numpy as np
#from replay_buffer import ReplayBuffer
from agent import Agent
# I copied the class from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py#L154
import numpy as np
import random
from collections import namedtuple, deque
import torch

from replay_buffer import ReplayBuffer
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): seed
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=(["state", "action", "reward", "next_state", "done"]))
    
    def add(self, x, action, reward, next_x, done):
        """Add a new experience to memory."""
        
        e = self.experience(x, action, reward, next_x, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
'''
class MultiAgent:
    def __init__(self, observation_state_size, action_state_size, hyperparameter, num_agents, seed):#todo remove seed, but use same seed in Agent + comment from solution
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_agents = num_agents
        # I got using single param for all hyperparameters from udacity code review for my previous project: P2 Continuous Control
        self.agents = [Agent(observation_state_size, action_state_size, hyperparameter, seed) for i in range(num_agents)]#todo remove seed, but use same seed in Agent + comment from solution
        self.sample_batch_size = hyperparameter['sample_batch_size']
        self.replay_buffer_size = hyperparameter['replay_buffer_size']
        self.replay_buffer = ReplayBuffer(action_state_size, self.replay_buffer_size, self.sample_batch_size, seed)
        self.gamma = hyperparameter['gamma']
        self.hyperparameter = hyperparameter
        self.action_state_size = action_state_size

    def reset(self):#todo rename to reset_noise, my name
        for i in range(self.num_agents):
            # reset noise
            # I got this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb
            self.agents[i].reset_noise

    def select_actions(self, all_agents_states, epsilon):
        #actions = np.zeros([self.num_agents, self.action_state_size])
        #for i in range(self.num_agents):
        #    actions[i, :] = self.agents[i].select_action(all_agents_states[i], epsilon)
        #print(f'their actions {actions}')
        actions = [self.agents[i].select_action(all_agents_states[i], epsilon) for i in range(self.num_agents)]
        actions = np.asarray(actions) #todo move back to collab_competition.py
        #print(f'my actions {my_actions}')
        return actions

    def add_to_buffer(self, all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones):
        # Udacity Honor Code: Code copied
        # I copied the below transformation of the observations
        # from here: https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/MADDPG.py#L44
        all_agents_states = np.concatenate(all_agents_states, axis=0)
        all_agents_next_states = np.concatenate(all_agents_next_states, axis=0)
        all_agents_actions = np.concatenate(all_agents_actions, axis=0)

        self.replay_buffer.add(all_agents_states, all_agents_actions, all_agents_rewards, all_agents_next_states, all_agents_dones)        

    # I copied the content of this method from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L78
    # and adjusted it for the multi agent part
    def learn(self, timestep):
        # only learn if enough data available
        # I copied this from here: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py#L60
        if(len(self.replay_buffer) > self.sample_batch_size):#todo >=
            # critic
            for num_agent, agent in enumerate(self.agents):
                # Udacity Honor Code: As mentioned in README, I had a bug in my implementation
                # add did not get a fresh sample for each agent, but used the same sample for both
                # Got the correct version from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                experiences = self.replay_buffer.sample()
                state, action, reward, next_state, done = experiences
                # Udacity Honor Code: As mentioned in README, I had a bug in the transformation of
                # the observations out of the replay buffer
                # Got the correct transformation from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                x, actions, rewards, next_x, dones = state, action, reward, next_state, done
                states = torch.chunk(x, 2, dim = 1)
                next_states = torch.chunk(next_x, 2, dim = 1)
                rewards = rewards[:,num_agent].reshape(rewards.shape[0],1)
                dones = dones[:,num_agent].reshape(dones.shape[0],1)
                
                # I got the implementation of updating the actor and critic from
                # the MADDPG-Lab implementation of the Physical Deception Problem, which is not public available (Udacity course material)
                # provided by udacity
                agent.critic_local.train()
                local_q_values = agent.critic_local(x, actions)

                # Udacity Honor Code: As mentioned in README, I called
                # actor_target not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                all_agents_next_actions = [self.agents[num_agent].actor_target(next_state_for_agent) for next_state_for_agent in next_states]
                target_actions = torch.cat(all_agents_next_actions, dim=1).to(self.device)  
                
                target_q_values = rewards + (self.gamma * agent.critic_target(next_x, target_actions) * (1 - dones))
                critic_loss = F.mse_loss(local_q_values, target_q_values)

                agent.critic_local_optimizer.zero_grad()
                critic_loss.backward()
                # I copied this from the course in project 2 Continuous Control 'Benchmark Implementation' where the udacity's benchmark implementation for the previous project
                # is described and some special settings are explicitly highlighted  
                torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
                agent.critic_local_optimizer.step()
                agent.critic_local.eval()

                #actor
                # Udacity Honor Code: As mentioned in README, I called
                # actor_local not only for the agent within this loop, but for all agents
                # I got the correct version and the following transformation of the actions
                # from a solution for this project here: https://github.com/and-buk/Udacity-DRLND/tree/master/p_collaboration_and_competition
                actions_local = [self.agents[num_agent].actor_local(state) for state in states]
                actions_local = torch.cat(actions_local, 1)

                actor_loss = -self.agents[num_agent].critic_local(x, actions_local).mean()
                self.agents[num_agent].actor_local_optimizer.zero_grad()
                actor_loss.backward()
                self.agents[num_agent].actor_local_optimizer.step()
                # todo udacity honor code, i made this at differen point, also add to readme
                self.agents[num_agent].soft_update(self.agents[num_agent].critic_target, self.agents[num_agent].critic_local, timestep)
                self.agents[num_agent].soft_update(self.agents[num_agent].actor_target, self.agents[num_agent].actor_local, timestep)
                
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