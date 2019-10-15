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
MATCH = 3
OFF = 4

IRRELEVANT = 0
RELEVANT = 1
ACTIVE = 2

NORACE = 0 
RACE = 1

class CostMDP(object):
    def __init__(self, alpha, gamma, T, mining_cost, epsilon=10e-6):
        # params
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.mining_cost = mining_cost
        self.epsilon = epsilon

        # game
        self.action_count = 4
        self.fork_count = 3
        self.race_count = 2
        self.state_count = (T + 1) * (T + 1) * self.fork_count * self.race_count

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
                for fork in range(self.fork_count):
                    for race in range(self.race_count):
                        self.state_mapping[(a, h, fork, race)] = count
                        self.states.append((a, h, fork, race))
                        count += 1
    
    def initMatrices(self):
        for _ in range(self.action_count):
            self.transitions.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.rewards.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            a, h, fork, race = self.states[state_index]

            # adopt
            self.transitions[ADOPT][state_index, self.state_mapping[0, 0, IRRELEVANT, NORACE]] = 1

            # override
            if a > h:
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h-1, 0, IRRELEVANT, NORACE]] = 1
                self.rewards[OVERRIDE][state_index, self.state_mapping[a-h-1, 0, IRRELEVANT, NORACE]] = h + 1
            else:
                self.transitions[OVERRIDE][state_index, 0] = 1
                self.rewards[OVERRIDE][state_index, 0] = -10000

            # mine
            if race == NORACE:
                if (fork != ACTIVE) and (a < self.T) and (h < self.T):
                    self.transitions[MINE][state_index, self.state_mapping[a+1, h, IRRELEVANT, NORACE]] = self.alpha
                    self.transitions[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = (1 - self.alpha) 
                    self.rewards[MINE][state_index, self.state_mapping[a+1, h, IRRELEVANT, NORACE]] = -1 * self.alpha * self.mining_cost
                    self.rewards[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = -1 * self.alpha * self.mining_cost        
                elif (fork == ACTIVE) and (a > h) and (h > 0) and (a < self.T) and (h < self.T):
                    self.transitions[MINE][state_index, self.state_mapping[a+1, h, ACTIVE, NORACE]] = self.alpha
                    self.transitions[MINE][state_index, self.state_mapping[a-h, 1, RELEVANT, NORACE]] = (1 - self.alpha) * self.gamma
                    self.transitions[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = (1 - self.alpha) * (1 - self.gamma)
                    self.rewards[MINE][state_index, self.state_mapping[a+1, h, ACTIVE, NORACE]] = -1 * self.alpha * self.mining_cost
                    self.rewards[MINE][state_index, self.state_mapping[a-h, 1, RELEVANT, NORACE]] = h - self.alpha * self.mining_cost
                    self.rewards[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = -1 * self.alpha * self.mining_cost
                else:
                    self.transitions[MINE][state_index, 0] = 1
                    self.rewards[MINE][state_index, 0] = -10000
            else:
                if (fork == RELEVANT) and (a >= h) and (h > 0) and (a < self.T) and (h < self.T):
                    self.transitions[MINE][state_index, self.state_mapping[a+1, h, ACTIVE, NORACE]] = self.alpha
                    self.transitions[MINE][state_index, self.state_mapping[a-h, 1, RELEVANT, NORACE]] = (1 - self.alpha) * self.gamma
                    self.transitions[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = (1 - self.alpha) * (1 - self.gamma)
                    self.rewards[MINE][state_index, self.state_mapping[a+1, h, ACTIVE, NORACE]] = -1 * self.alpha * self.mining_cost
                    self.rewards[MINE][state_index, self.state_mapping[a-h, 1, RELEVANT, NORACE]] = h - self.alpha * self.mining_cost
                    self.rewards[MINE][state_index, self.state_mapping[a, h+1, RELEVANT, NORACE]] = -1 * self.alpha * self.mining_cost
                else:
                    self.transitions[MINE][state_index, 0] = 1
                    self.rewards[MINE][state_index, 0] = -10000
                
            # match 
            if (fork == RELEVANT) and (a >= h) and (h > 0) and (a < self.T) and (h < self.T):
                self.transitions[MATCH][state_index, self.state_mapping[a, h, fork, RACE]] = 1
            else:
                self.transitions[MATCH][state_index, 0] = 1
                self.rewards[MATCH][state_index, 0] = -10000

            # # off
            # if (fork != ACTIVE) and (h < self.T):
            #     self.transitions[OFF][state_index, self.state_mapping[a, h+1, fork]] = 1
            # elif (fork == ACTIVE) and (a > h) and (h > 0) and (h < self.T):
            #     self.transitions[OFF][state_index, self.state_mapping[a-h, 1, RELEVANT]] = self.gamma
            #     self.transitions[OFF][state_index, self.state_mapping[a, h+1, RELEVANT]] = 1-self.gamma
            #     self.rewards[OFF][state_index, self.state_mapping[a-h, 1, RELEVANT]] = h
            #     self.rewards[OFF][state_index, self.state_mapping[a, h+1, RELEVANT]] = 0
            # else:
            #     self.transitions[OFF][state_index, 0] = 1
            #     self.rewards[OFF][state_index, 0] = -10000
            
    def getOptPolicy(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, self.rewards, self.epsilon/8)
        rvi.run()
        print(rvi.average_reward)
        return rvi.policy

    def printPolicy(self, policy):    
        results = ''
        for a in range(9):
            for h in range(9):
                for fork in range(3):
                    for race in range(2):
                        state_index = self.state_mapping[(a, h, fork, race)]
                        action = policy[state_index]
                        if action == 0:
                            results += 'a'
                        elif action == 1:
                            results += 'o'
                        elif action == 2:
                            results += 'w'
                        elif action == 3:
                            results += 'm'
                        elif action == 4:
                            results += 'f'
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
    alpha = 0.4
    gamma = 0.5
    T = 8
    mining_cost = 0.5
    cost_mdp = CostMDP(alpha = alpha, gamma = gamma, T = T, mining_cost = mining_cost)
    cost_mdp.initMDPHelpers()
    cost_mdp.initMatrices()
    cost_mdp.populateMatrices()
    # print(np.sum(cost_mdp.transitions[], 1))
    policy = cost_mdp.getOptPolicy()
    cost_mdp.printPolicy(policy)
