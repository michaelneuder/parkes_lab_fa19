import numpy as np
np.random.seed(0)

class Environment(object):
    def __init__(self, mining_powers, gammas, T):
        # relative mining strengths.
        self.mining_powers = mining_powers
        self.gammas = gammas
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
        self.chain = self.chain[:self.starting_points[player_index]]
        new_blocks = str(player_index) * self.hidden_lengths[player_index]
        self.chain += new_blocks
        self.starting_points[player_index] = len(self.chain)
        self.hidden_lengths[player_index] = 0
    
    def match(self, player_index):
        # \alpha, \gamma * (1 - \alpha), (1 - \gamma) * (1 - \alpha)
        new_probs = [self.mining_powers[1], self.gammas[1] * self.mining_powers[0], self.gammas[0] * self.mining_powers[0]]
        next_block = np.random.choice(np.arange(len(new_probs)), p = new_probs)
        # print(self.chain)
        # print('match: new block', next_block)
        if next_block == 0:
            self.hidden_lengths[1] += 1
            self.override(1)
        elif next_block == 1:
            self.hidden_lengths[1] += 1
            self.override(1)
            self.chain = self.chain[:-1]
            self.chain += '0'
        else:
            self.chain += '0'
            self.adopt(1)
        # print(self.chain)

    def getState(self, player_index):
        return (self.hidden_lengths[player_index], len(self.chain)-self.starting_points[player_index])

if __name__ == "__main__":
    powers = [0.55, 0.45]
    gammas = [0.5, 0.5]
    env = Environment(powers, gammas, T=9)
    chain = ''
    for _ in range(1000):
        chain += str(env.getNextBlockWinner())
    print('p0', chain.count('0'), chain.count('0')/len(chain))
    print('p1', chain.count('1'), chain.count('1')/len(chain))
    print('p2', chain.count('2'), chain.count('2')/len(chain))
    