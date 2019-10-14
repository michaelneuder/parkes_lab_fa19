import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
import scipy.sparse as ss
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

ADOPT = 0
OVERRIDE = 1
MINE = 2

class CostReducedMDP(object):
    def __init__(self, alpha, T, mining_cost, epsilon=10e-6):
        # params
        self.alpha = alpha
        self.T = T
        self.mining_cost = mining_cost
        self.epsilon = epsilon

        # game
        self.action_count = 3
        self.state_count = (T + 1) * (T + 1)

        # mdp helpers
        self.state_mapping = {}
        self.states = []

        # matrices
        self.transitions = []
        self.rewards = []

    def initMDPHelpers(self):
        count = 0
        for a in range(self.T+1):
            for h in range(self.T+1):
                self.state_mapping[(a, h)] = count
                self.states.append((a, h))
                count += 1

    def initMatrices(self):
        for _ in range(self.action_count):
            self.transitions.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.rewards.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            a, h = self.states[state_index]

            # adopt
            self.transitions[ADOPT][state_index, self.state_mapping[0, 0]] = 1

            # override
            if a > h:
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h-1, 0]] = 1
                self.rewards[OVERRIDE][state_index, self.state_mapping[a-h-1, 0]] = h + 1
            else:
                self.transitions[OVERRIDE][state_index, 0] = 1
                self.rewards[OVERRIDE][state_index, 0] = -10000

            # mine 
            if (a < self.T) and (h < self.T):
                self.transitions[MINE][state_index, self.state_mapping[a+1, h]] = self.alpha
                self.transitions[MINE][state_index, self.state_mapping[a, h+1]] = (1 - self.alpha) 
                self.rewards[MINE][state_index, self.state_mapping[a+1, h]] = -1 * self.alpha * self.mining_cost
                self.rewards[MINE][state_index, self.state_mapping[a, h+1]] = -1 * self.alpha * self.mining_cost        
            else:
                self.transitions[MINE][state_index, 0] = 1
                self.rewards[MINE][state_index, 0] = -10000
                
    
    def getOptPolicy(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, self.rewards, self.epsilon/8)
        rvi.run()
        return rvi.policy
    
    def getOptValue(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, self.rewards, self.epsilon/8)
        rvi.run()
        return rvi.average_reward

    def printPolicy(self, policy):    
        results = ''
        for a in range(9):
            for h in range(9):
                state_index = self.state_mapping[(a, h)]
                action = policy[state_index]
                if action == 0:
                    results += 'a'
                elif action == 1:
                    results += 'o'
                elif action == 2:
                    results += 'w'
                else:
                    raise RuntimeError('invalid action')
                results += ' & '
            results += '\\\\ \n'
        print(results)
    
    def getAction(self, policy, state):
        state_index = self.state_mapping[state]
        return policy[state_index]

    def solveWithPolicy(self):
        self.initMDPHelpers()
        self.initMatrices()
        self.populateMatrices()
        return self.getOptPolicy()
    
if __name__ == "__main__":
    cost_reduced_mdp = CostReducedMDP(alpha = 0.4, T = 8, mining_cost = 0.5)
    cost_reduced_mdp.initMDPHelpers()
    cost_reduced_mdp.initMatrices()
    cost_reduced_mdp.populateMatrices()
    policy = cost_reduced_mdp.getOptPolicy()
    cost_reduced_mdp.printPolicy(policy)
