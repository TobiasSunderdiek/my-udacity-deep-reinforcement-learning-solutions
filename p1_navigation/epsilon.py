class Epsilon:

    def __init__(self, epsilon_start, epsilon_max_decay_to, max_timesteps_episode):
        self.epsilon_start = epsilon_start
        self.epsilon_max_decay_to = epsilon_max_decay_to
        self.max_timesteps_episode = max_timesteps_episode

    def calculate_for(self, timestep):
        return max(timestep * self.epsilon_start/self.max_timesteps_episode, self.epsilon_max_decay_to)