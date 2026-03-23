import numpy as np

class DeepQAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actions = [a for a in range(self.action_size)]

    def act(self, state):
        return np.random.choice(self.actions)
