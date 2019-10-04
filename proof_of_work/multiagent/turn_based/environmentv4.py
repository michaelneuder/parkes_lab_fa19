import numpy as np
np.random.seed(0)

class Environment(object):
    def __init__(self, mining_powers, T):
        # relative mining strengths.
        self.mining_powers = mining_powers
        self.num_miners = len(mining_powers)
        
        # termination parameters
        self.T = T

        # chain variables
        self.chain = ''
        self.starting_points = np.zeros(self.num_miners, dtype=np.int64)
        self.hidden_lengths = np.zeros(self.num_miners, dtype=np.int64)

    def getNextBlockWinner(self):
        winner = np.random.choice(np.arange(len(self.mining_powers)), p=self.mining_powers)
        self.hidden_lengths[winner] += 1
        return winner
    
    def adopt(self, player_index):
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
    
    def override(self, player_index):
        # chop chain to proper length
        self.chain = self.chain[:self.starting_points[player_index] + 1]
        new_blocks = '{}'.format(player_index) * self.hidden_lengths[player_index]
        self.chain += new_blocks
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
    
    def getState(self, player_index):
        return (self.hidden_lengths[player_index], len(self.chain)-self.starting_points[player_index])

if __name__ == "__main__":
    powers = [0.55, 0.45]
    env = Environment(powers, T=9)
    chain = ''
    for _ in range(1000):
        chain += str(env.getNextBlockWinner())
    print('p0', chain.count('0'), chain.count('0')/len(chain))
    print('p1', chain.count('1'), chain.count('1')/len(chain))
    print('p2', chain.count('2'), chain.count('2')/len(chain))
    