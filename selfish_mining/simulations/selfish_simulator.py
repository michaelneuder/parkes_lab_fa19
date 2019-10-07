import copy
import matplotlib.pyplot as plt
import numpy as np
import progressbar

class Simulator(object):
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.public_chain = ''
        self.private_chain = ''
        self.private_branch_length = 0
    
    def simulate(self):
        bar = progressbar.ProgressBar()
        for _ in bar(range(100000)):
            selfish_block = np.random.choice([1, 0], p=[self.alpha, 1-self.alpha])
            if selfish_block:
                prev_diff = len(self.private_chain) - len(self.public_chain)
                self.private_chain += 's'
                self.private_branch_length += 1
                
                # Selfish miner won the 1 to 1 tie and publishes entire private chain.
                if (prev_diff == 0) and (self.private_branch_length == 2):
                    self.public_chain = self.private_chain
                    self.private_branch_length = 0
            
            else:
                prev_diff = len(self.private_chain) - len(self.public_chain)
                self.public_chain += 'h'
                
                # Honest miners win race.
                if prev_diff == 0:
                    self.private_chain = self.public_chain
                    self.private_branch_length = 0
                
                # Selfish miners were 1 ahead and honest caught them.
                elif prev_diff == 1:
                    # Probability of winning is all selfish miners and all honest miners who
                    # use the selfish block to mine on.
                    total_compute_power = self.alpha + (1 - self.alpha)*self.gamma
                    if np.random.uniform() < total_compute_power:
                        self.public_chain = self.private_chain
                
                # Selfish was 2 ahead and now published entire private.
                elif prev_diff == 2:
                    self.public_chain = self.private_chain
                    self.private_branch_length = 0
        
        # At end if selfish have lead they publish them all.
        if self.private_branch_length > 0:
            # print(self.public_chain)
            self.public_chain = copy.copy(self.private_chain)

if __name__ == "__main__":
    simulator = Simulator(0.35, 0)
    simulator.simulate()
    print(simulator.public_chain.count('s') / len(simulator.public_chain))