import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

ADOPT = 0
OVERRIDE = 1
WAIT = 2


class OriginalReducedMDP(object):
    def __init__(self, alpha, T, epsilon=10e-6): 
        # params
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        
        # game
        self.action_count = 3
        self.state_count = (T+1) * (T+1)

        # mdp helpers
        self.state_mapping = {}
        self.states = []

        # matrices
        self.transitions = []
        self.reward_selfish = []
        self.reward_honest = []

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
            self.reward_honest.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            a, h = self.states[state_index]
            
            # adopt transitions
            self.transitions[ADOPT][state_index, self.state_mapping[1, 0]] = self.alpha
            self.transitions[ADOPT][state_index, self.state_mapping[0, 1]] = 1 - self.alpha
            # adopt rewards
            self.reward_honest[ADOPT][state_index, self.state_mapping[1, 0]] = h
            self.reward_honest[ADOPT][state_index, self.state_mapping[0, 1]] = h

            # override
            if a > h:
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h, 0]] = self.alpha
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h, 0]] = h+1
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h-1, 1]] = 1 - self.alpha
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h-1, 1]] = h+1
            else:
                self.transitions[OVERRIDE][state_index, 0] = 1
                self.reward_honest[OVERRIDE][state_index, 0] = 10000

            # wait transitions
            if (a < self.T) and (h < self.T):
                self.transitions[WAIT][state_index, self.state_mapping[a+1, h]] = self.alpha
                self.transitions[WAIT][state_index, self.state_mapping[a, h+1]] = 1 - self.alpha
            else:
                self.transitions[WAIT][state_index, 0] = 1
                self.reward_honest[WAIT][state_index, 0] = 10000

    def getRhoBounds(self):
        low = 0; high = 1
        while (high - low) > self.epsilon / 8:
            rho = (low + high) / 2
            print(low, high, rho)
            total_reward = []
            for i in range(self.action_count):
                total_reward.append((1-rho)*self.reward_selfish[i] - rho*self.reward_honest[i])
            rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, total_reward, self.epsilon/8)
            rvi.run()
            if rvi.average_reward > 0:
                low = rho
            else:
                high = rho
        opt_policy = rvi.policy
        print('alpha: ', self.alpha, 'lower bound reward:', rho)
        self.processPolicy(opt_policy)
        
    def processPolicy(self, policy):
        results = ''
        for a in range(9):
            results += '{} & '.format(a)
            for h in range(9):
                state_index = self.state_mapping[(a, h)]
                action = policy[state_index]
                assert(action in [0, 1, 2])
                if action == 0:
                    results += 'a'
                elif action == 1:
                    results += 'o'
                else:
                    results += 'w'
                results += ' & '
            results = results[:-2]
            results += '\\\\ \n'
        print(results)

if __name__ == "__main__":
    alpha = 0.4
    T = 8
    original_mdp = OriginalReducedMDP(alpha=alpha, T=T, epsilon=10e-5)
    original_mdp.initMDPHelpers()
    original_mdp.initMatrices()
    original_mdp.populateMatrices()
    original_mdp.getRhoBounds()
