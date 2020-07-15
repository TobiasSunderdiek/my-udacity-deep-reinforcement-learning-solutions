class Epsilon:

    def __init__(self, epsilon_start, epsilon_decay_rate, epsilon_max_decay_to):
        self.epsilon_start = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_max_decay_to = epsilon_max_decay_to

    def calculate_for(self, timestep):
        return max(self.epsilon_start * (self.epsilon_decay_rate ** timestep), self.epsilon_max_decay_to)