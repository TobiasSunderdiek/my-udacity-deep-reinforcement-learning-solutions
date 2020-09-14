import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic

seed = 0
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 250        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
OU_THETA = 0.15         # how "strongly" the system reacts to perturbations
OU_SIGMA = 0.2          # the variation or the size of the noise
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

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
        self.experience = namedtuple("Experience", field_names=["x", "action", "reward", "next_x", "done"])
    
    def add(self, x, action, reward, next_x, done):
        """Add a new experience to memory."""
        
        # Join a sequence of agents's states, next states and actions along columns
        x = np.concatenate(x, axis=0)
        next_x = np.concatenate(next_x, axis=0)
        action = np.concatenate(action, axis=0)
        
        e = self.experience(x, action, reward, next_x, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        x = torch.from_numpy(np.vstack([e.x for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_x = torch.from_numpy(np.vstack([e.next_x for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (x, actions, rewards, next_x, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# Initialize experience replay buffer to recording experiences of all agents
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed = seed)
class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, hyperparameter, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, hyperparameter, self.seed).to(device)
        self.actor_target = Actor(state_size, action_size, hyperparameter, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*2, action_size*2, hyperparameter, self.seed).to(device)
        self.critic_target = Critic(state_size*2, action_size*2, hyperparameter, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process for action exploration
        self.noise = OUNoise(action_size, self.seed)
                 
    def step(self, num_agent):
        """Use random sample from buffer to learn."""
        # If enough samples are available in memory, get random subset and learn 
        if len(memory) > BATCH_SIZE:
            experiences = memory.sample()
            self.learn(experiences, GAMMA, num_agent)        
      
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval() # Set the local policy in evaluation mode
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train() # Set the local policy in training mode
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, num_agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (x, a, r, x', done) tuples 
            gamma (float): discount factor
        """ 
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
        next_actions = [self.actor_target(n_s) for n_s in next_states]
        target_actions = torch.cat(next_actions, dim=1).to(device)  
        Q_targets_next = self.critic_target(next_x, target_actions)        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))        
        # Compute critic loss
        Q_expected = self.critic_local(x, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        actions_pred = [self.actor_local(s) for s in states]        
        actions_pred_ = torch.cat(actions_pred, dim=1).to(device)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(x, actions_pred_).mean()        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()        
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()  
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state 
    
class MultiAgent:
    """Interaction between multiple agents in common environment"""
    def __init__(self, state_size, action_size, hyperparameter, num_agents, seed):
        self.state_size = state_size
        self.action_size = action_size        
        self.num_agents = num_agents
        self.seed = seed
        self.agents = [Agent(self.state_size, self.action_size, hyperparameter, self.seed) for x in range(self.num_agents)]

    def add_to_buffer(self, x, actions, rewards, next_x, dones):
        memory.add(x, actions, rewards, next_x, dones)

    
    def learn(self, timestep):
        for num_agent, agent in enumerate(self.agents):
            agent.step(num_agent)
    def select_actions(self, all_agents_states, current_epsilon):
        return self.act(all_agents_states, True)

    def step(self, x, actions, rewards, next_x, dones):
        """Save experiences in replay memory and learn."""
        # Save experience in replay memory
        memory.add(x, actions, rewards, next_x, dones)
        
        for num_agent, agent in enumerate(self.agents):
            agent.step(num_agent)

    def act(self, states, add_noise=True):
        """Agents perform actions according to their policy."""
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions
    
    def reset(self):        
        for agent in self.agents:
            agent.reset()

    def save_model(self):
        """Save learnable model's parameters of each agent."""
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index + 1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index + 1))
            
    def load_model(self):
        """Load learnable model's parameters of each agent."""
        for index, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor.pth'.format(index + 1)))
            agent.critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic.pth'.format(index + 1)))