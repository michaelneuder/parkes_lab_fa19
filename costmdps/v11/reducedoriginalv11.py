import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class ReducedMDP(object):
    def __init__(self, alpha, T, epsilon=10e-6): 
        # params
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        
        # game
        self.action_count = 3
        self.state_count = (T + 1) * (T + 1)

        # mdp helpers
        self.state_mapping = {}
        self.states = []

        # matrices
        self.transitions = []
        self.reward_selfish = []

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
            self.reward_selfish.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            a, h = self.states[state_index]
            
            # adopt
            self.transitions[ADOPT][state_index, self.state_mapping[1, 0]] = self.alpha
            self.transitions[ADOPT][state_index, self.state_mapping[0, 1]] = 1 - self.alpha
            
            # override
            if a > h:
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h, 0]] = self.alpha
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h-1, 1]] = 1 - self.alpha
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h, 0]] = h + 1
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h-1, 1]] = h + 1
            else:
                self.transitions[OVERRIDE][state_index, 0] = 1
                self.reward_selfish[OVERRIDE][state_index, 0] = -10000

            # wait
            if (a < self.T) and (h < self.T):
                self.transitions[WAIT][state_index, self.state_mapping[a+1, h]] = self.alpha
                self.transitions[WAIT][state_index, self.state_mapping[a, h+1]] = 1 - self.alpha
            else:
                self.transitions[WAIT][state_index, 0] = 1
                self.reward_selfish[WAIT][state_index, 0] = -10000

    def getOptPolicy(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, self.reward_selfish, self.epsilon/8)
        rvi.run()
        opt_policy = rvi.policy
        print(rvi.average_reward)
        return opt_policy
        
    def printPolicy(self, policy):
        results = ''
        for a in range(9):
            results += str(a) + ' & '
            for h in range(9):
                state_index = self.state_mapping[(a, h)]
                action = policy[state_index]
                if action == 0:
                    results += 'a'
                elif action == 1:
                    results += 'o'
                elif action == 2:
                    results += 'w'
                elif action == 3:
                    results += 'm'
                else:
                    raise RuntimeError('invalid action')
                results += ' & '
            results = results[:-2]
            results += '\\\\ \n'
        print(results)
    
    def solveWithPolicy(self):
        self.initMDPHelpers()
        self.initMatrices()
        self.populateMatrices()
        return self.getOptPolicy()
    
    def getAction(self, policy, state):
        state_index = self.state_mapping[state]
        return policy[state_index]

if __name__ == "__main__":
    alpha = 0.4
    gamma = 0.5
    T = 8
    original_mdp = ReducedMDP(alpha=alpha, T=T, epsilon=10e-5)
    original_mdp.initMDPHelpers()
    original_mdp.initMatrices()
    original_mdp.populateMatrices()
    opt_policy = original_mdp.getOptPolicy()
    original_mdp.printPolicy(opt_policy)
    print(np.reshape(opt_policy, (T+1, T+1)))
