import numpy as np
np.random.seed(0)

class Environment(object):
    def __init__(self, mining_powers, T, max_blocks=1000):
        # relative mining strengths.
        self.mining_powers = mining_powers
        self.honest_power = 1 - sum(mining_powers)
        self.num_miners = len(mining_powers)
        
        # termination parameters
        self.T = T
        self.max_blocks = max_blocks

        # chain variables
        self.chain = ''
        self.starting_points = np.zeros(self.num_miners, dtype=np.int64)
        self.hidden_lengths = np.zeros(self.num_miners, dtype=np.int64)

    def getNextBlockWinner(self):
        return np.random.choice(np.arange(len(self.mining_powers)), p=self.mining_powers)

