import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

ADOPT = 0
OVERRIDE = 1
WAIT = 2
MATCH = 3

IRRELEVANT = 0
RELEVANT = 1
ACTIVE = 2

class OriginalMDP(object):
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
        self.state_count = (T + 1) * (T + 1) * self.fork_count

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
                for fork in range(self.fork_count):
                    self.state_mapping[(a, h, fork)] = count
                    self.states.append((a, h, fork))
                    count += 1
    
    def initMatrices(self):
        for _ in range(self.action_count):
            self.transitions.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.reward_selfish.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.reward_honest.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def populateMatrices(self):
        for state_index in range(self.state_count):
            a, h, fork = self.states[state_index]
            
            # adopt transitions
            self.transitions[ADOPT][state_index, self.state_mapping[1, 0, IRRELEVANT]] = self.alpha
            self.transitions[ADOPT][state_index, self.state_mapping[0, 1, IRRELEVANT]] = 1 - self.alpha
            # adopt rewards
            self.reward_honest[ADOPT][state_index, self.state_mapping[1, 0, IRRELEVANT]] = h
            self.reward_honest[ADOPT][state_index, self.state_mapping[0, 1, IRRELEVANT]] = h
            self.reward_selfish[ADOPT][state_index, self.state_mapping[1, 0, IRRELEVANT]] = -1 * self.alpha * self.mining_cost 
            self.reward_selfish[ADOPT][state_index, self.state_mapping[0, 1, IRRELEVANT]] = -1 * self.alpha * self.mining_cost


            # override
            if a > h:
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h, 0, IRRELEVANT]] = self.alpha
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h, 0, IRRELEVANT]] = h+1 - self.alpha * self.mining_cost
                self.transitions[OVERRIDE][state_index, self.state_mapping[a-h-1, 1, RELEVANT]] = 1 - self.alpha
                self.reward_selfish[OVERRIDE][state_index, self.state_mapping[a-h-1, 1, RELEVANT]] = h+1 - self.alpha * self.mining_cost
            else:
                self.transitions[OVERRIDE][state_index, 0] = 1
                self.reward_honest[OVERRIDE][state_index, 0] = 10000

            # wait
            if (fork != ACTIVE) and (a < self.T) and (h < self.T):
                self.transitions[WAIT][state_index, self.state_mapping[a+1, h, IRRELEVANT]] = self.alpha
                self.transitions[WAIT][state_index, self.state_mapping[a, h+1, RELEVANT]] = 1 - self.alpha
                self.reward_selfish[WAIT][state_index, self.state_mapping[a+1, h, IRRELEVANT]] = -1 * self.alpha * self.mining_cost
                self.reward_selfish[WAIT][state_index, self.state_mapping[a, h+1, RELEVANT]] = -1 * self.alpha * self.mining_cost
            elif (fork == ACTIVE) and (a > h) and (h > 0) and (a < self.T) and (h < self.T): 
                self.transitions[WAIT][state_index, self.state_mapping[a+1, h, ACTIVE]] = self.alpha
                self.reward_selfish[WAIT][state_index, self.state_mapping[a+1, h, ACTIVE]] = -1 * self.alpha * self.mining_cost
                self.transitions[WAIT][state_index, self.state_mapping[a-h, 1, RELEVANT]] = self.gamma*(1-self.alpha)
                self.reward_selfish[WAIT][state_index, self.state_mapping[a-h, 1, RELEVANT]] = h - self.alpha * self.mining_cost
                self.transitions[WAIT][state_index, self.state_mapping[a, h+1, RELEVANT]] = (1-self.gamma)*(1-self.alpha)
                self.reward_selfish[WAIT][state_index, self.state_mapping[a, h+1, RELEVANT]] = -1 * self.alpha * self.mining_cost
            else:
                self.transitions[WAIT][state_index, 0] = 1
                self.reward_honest[WAIT][state_index, 0] = 10000

            # match
            if (fork == RELEVANT) and (a >= h) and (h > 0) and (a < self.T) and (h < self.T):
                self.transitions[MATCH][state_index, self.state_mapping[a+1, h, ACTIVE]] = self.alpha
                self.reward_selfish[MATCH][state_index, self.state_mapping[a+1, h, ACTIVE]] = -1 * self.alpha * self.mining_cost
                self.transitions[MATCH][state_index, self.state_mapping[a-h, 1, RELEVANT]] = self.gamma*(1-self.alpha)
                self.reward_selfish[MATCH][state_index, self.state_mapping[a-h, 1, RELEVANT]] = h - self.alpha * self.mining_cost
                self.transitions[MATCH][state_index, self.state_mapping[a, h+1, RELEVANT]] = (1-self.gamma)*(1-self.alpha)
                self.reward_selfish[MATCH][state_index, self.state_mapping[a, h+1, RELEVANT]] = -1 * self.alpha * self.mining_cost
            else:
                self.transitions[MATCH][state_index, 0] = 1
                self.reward_honest[MATCH][state_index, 0] = 10000

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
        self.printPolicy(opt_policy)
        
    def printPolicy(self, policy):    
        results = ''
        for a in range(9):
            for h in range(9):
                for fork in range(3):
                    state_index = self.state_mapping[(a, h, fork)]
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
            results += '\\\\ \n'
        print(results)

if __name__ == "__main__":
    alpha = 0.4
    gamma = 0.5
    T = 8
    mining_cost = 0.82
    original_mdp = OriginalMDP(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost, epsilon=10e-5)
    original_mdp.initMDPHelpers()
    original_mdp.initMatrices()
    original_mdp.populateMatrices()
    original_mdp.getRhoBounds()
