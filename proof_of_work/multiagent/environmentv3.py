import copy
import numpy as np
np.random.seed(0)

class Environment(object):
    def __init__(self, mining_powers, T, max_blocks=1000, difficulty=0.01):
        # relative mining strengths.
        self.mining_powers = mining_powers
        self.honest_power = 1 - sum(mining_powers)
        self.num_miners = len(mining_powers)
        
        # termination parameters
        self.T = T
        self.max_blocks = max_blocks

        # chain variables
        self.chain = 'h'
        self.starting_points = np.zeros(self.num_miners, dtype=np.int64)
        self.hidden_lengths = np.zeros(self.num_miners, dtype=np.int64)
        self.difficulty = difficulty

    def getState(self, player_index):
        return (len(self.chain), self.starting_points[player_index], self.hidden_lengths[player_index])

    def honestMine(self):
        # chance that the honest network mines a block
        if np.random.uniform() < (self.honest_power * self.difficulty) :
            self.chain += 'h'
    
    def playerMine(self, player_index):
        # chance the player mines a block
        if np.random.uniform()  < self.mining_powers[player_index]*self.difficulty:
            self.hidden_lengths[player_index] += 1
        return (len(self.chain), self.starting_points[player_index], self.hidden_lengths[player_index])
    
    def playerOverride(self, player_index):
        self.chain = self.chain[:self.starting_points[player_index] + 1]
        new_blocks = '{}'.format(player_index) * self.hidden_lengths[player_index]
        self.chain += new_blocks
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0

    def playerAdopt(self, player_index):
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
        return(len(self.chain), self.starting_points[player_index], self.hidden_lengths[player_index])


