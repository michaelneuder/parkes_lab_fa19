import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

PASS = 0
ENDORSE = 1
BAKE = 2
BOTH = 3
ACTIONS = ['PASS', 'ENDORSE', 'BAKE', 'BOTH']

class PosMDP(object):
    def __init__(self, alpha, beta): 
        # params
        self.alpha = alpha
        self.beta = beta

        # game
        self.action_count = 4
        self.state_count = 2 * 2 * 2

        # mdp helpers
        self.state_mapping = {}
        self.states = []

        # matrices
        self.transitions = []
        self.rewards = []

    def initMDPHelpers(self):
        count = 0
        for B in range(2):
            for T in range(2):
                for E in range(2):
                    self.state_mapping[(B, T, E)] = count
                    self.states.append((B, T, E))
                    count += 1
    
    def initMatrices(self):
        for _ in range(self.action_count):
            self.transitions.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.rewards.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            B, T, E = self.states[state_index]
            
            # pass
            if (T == 0) and (B == 1):
                self.transitions[PASS][state_index, self.state_mapping[1, 1, E]] = 1
            else:
                self.transitions[PASS][state_index, self.state_mapping[0, 0, 0]] = (1-self.alpha) * (1-self.beta)
                self.transitions[PASS][state_index, self.state_mapping[0, 0, 1]] = (1-self.alpha) * self.beta
                self.transitions[PASS][state_index, self.state_mapping[1, 0, 0]] = self.alpha * (1-self.beta)
                self.transitions[PASS][state_index, self.state_mapping[1, 0, 1]] = self.alpha * self.beta

            # endorse
            if (E == 1):
                self.transitions[ENDORSE][state_index, self.state_mapping[0, 0, 0]] = (1-self.alpha) * (1-self.beta)
                self.transitions[ENDORSE][state_index, self.state_mapping[0, 0, 1]] = (1-self.alpha) * self.beta
                self.transitions[ENDORSE][state_index, self.state_mapping[1, 0, 0]] = self.alpha * (1-self.beta)
                self.transitions[ENDORSE][state_index, self.state_mapping[1, 0, 1]] = self.alpha * self.beta
                self.rewards[ENDORSE][state_index, self.state_mapping[0, 0, 0]] = 2
                self.rewards[ENDORSE][state_index, self.state_mapping[0, 0, 1]] = 2
                self.rewards[ENDORSE][state_index, self.state_mapping[1, 0, 0]] = 2
                self.rewards[ENDORSE][state_index, self.state_mapping[1, 0, 1]] = 2
            else:
                self.transitions[ENDORSE][state_index, self.state_mapping[0, 0, 0]] = 1
                self.rewards[ENDORSE][state_index, self.state_mapping[0, 0, 0]] = -100

            # bake
            if (B == 1):
                self.transitions[BAKE][state_index, self.state_mapping[0, 0, 0]] = (1-self.alpha) * (1-self.beta)
                self.transitions[BAKE][state_index, self.state_mapping[0, 0, 1]] = (1-self.alpha) * self.beta
                self.transitions[BAKE][state_index, self.state_mapping[1, 0, 0]] = self.alpha * (1-self.beta)
                self.transitions[BAKE][state_index, self.state_mapping[1, 0, 1]] = self.alpha * self.beta
                self.rewards[BAKE][state_index, self.state_mapping[0, 0, 0]] = 16
                self.rewards[BAKE][state_index, self.state_mapping[0, 0, 1]] = 16
                self.rewards[BAKE][state_index, self.state_mapping[1, 0, 0]] = 16
                self.rewards[BAKE][state_index, self.state_mapping[1, 0, 1]] = 16
            else:
                self.transitions[BAKE][state_index, self.state_mapping[0, 0, 0]] = 1
                self.rewards[BAKE][state_index, self.state_mapping[0, 0, 0]] = -100

            # both
            if (B == 1) and (E == 1):
                self.transitions[BOTH][state_index, self.state_mapping[0, 0, 0]] = (1-self.alpha) * (1-self.beta)
                self.transitions[BOTH][state_index, self.state_mapping[0, 0, 1]] = (1-self.alpha) * self.beta
                self.transitions[BOTH][state_index, self.state_mapping[1, 0, 0]] = self.alpha * (1-self.beta)
                self.transitions[BOTH][state_index, self.state_mapping[1, 0, 1]] = self.alpha * self.beta
                self.rewards[BOTH][state_index, self.state_mapping[0, 0, 0]] = 18
                self.rewards[BOTH][state_index, self.state_mapping[0, 0, 1]] = 18
                self.rewards[BOTH][state_index, self.state_mapping[1, 0, 0]] = 18
                self.rewards[BOTH][state_index, self.state_mapping[1, 0, 1]] = 18
            else:
                self.transitions[BOTH][state_index, self.state_mapping[0, 0, 0]] = 1
                self.rewards[BOTH][state_index, self.state_mapping[0, 0, 0]] = -100


    def getOptPolicy(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, self.rewards, float(1e-5))
        rvi.run()
        opt_policy = rvi.policy
        print(rvi.average_reward)
        return opt_policy
        
    def printPolicy(self, policy):
        for B in range(2):
            for T in range(2):
                for E in range(2):
                    state_index = self.state_mapping[(B,T,E)]
                    action = policy[state_index]
                    print('B={}, T={}, E={} -- action={}'.format(B,T,E,ACTIONS[action]))
    
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
    beta = 0.1
    original_mdp = PosMDP(alpha=alpha, beta=beta)
    original_mdp.initMDPHelpers()
    original_mdp.initMatrices()
    original_mdp.populateMatrices()
    opt_policy = original_mdp.getOptPolicy()
    print(opt_policy)
    original_mdp.printPolicy(opt_policy)

