import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

class SelfishMDP(object):
    def __init__(self, alpha, T, epsilon): 
        self.alpha = alpha
        self.T = T
        self.state_count = (T+1) * (T+1)
        self.epsilon = epsilon
        self.initMDPHelpers()
        print('initializing matrices...')
        self.initMatrices()
        print('done!')
        print('writing to matrices...')
        self.writeMatrixData()
        print('done!')
    
    def initMDPHelpers(self):
        self.action_count = 3
        self.adopt = 0; self.override = 1; self.wait = 2
        # generate a state to integer mapping and list of states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(self.T+1):
            for h in range(self.T+1):
                    self.state_mapping[(a, h)] = count
                    self.states.append((a, h))
                    count += 1
    
    def initMatrices(self):
        # transition and reward matrices
        self.transitions = []; self.reward_selfish = []; self.reward_honest = []
        for _ in range(self.action_count):
            self.transitions.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.reward_selfish.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
            self.reward_honest.append(ss.csr_matrix(np.zeros(shape=(self.state_count, self.state_count))))
    
    def writeMatrixData(self):
       # writing transition and reward data 
        for state_index in range(self.state_count):
            a, h = self.states[state_index]
            
            # adopt transitions
            self.transitions[self.adopt][state_index, self.state_mapping[1, 0]] = self.alpha
            self.transitions[self.adopt][state_index, self.state_mapping[0, 1]] = 1 - self.alpha
            # adopt rewards
            self.reward_honest[self.adopt][state_index, self.state_mapping[1, 0]] = h
            self.reward_honest[self.adopt][state_index, self.state_mapping[0, 1]] = h

            # override
            if a > h:
                self.transitions[self.override][state_index, self.state_mapping[a-h, 0]] = self.alpha
                self.reward_selfish[self.override][state_index, self.state_mapping[a-h, 0]] = h+1
                self.transitions[self.override][state_index, self.state_mapping[a-h-1, 1]] = 1 - self.alpha
                self.reward_selfish[self.override][state_index, self.state_mapping[a-h-1, 1]] = h+1
            else:
                self.transitions[self.override][state_index, 0] = 1
                self.reward_honest[self.override][state_index, 0] = 10000

            # wait transitions
            if (a < self.T) and (h < self.T):
                self.transitions[self.wait][state_index, self.state_mapping[a+1, h]] = self.alpha
                self.transitions[self.wait][state_index, self.state_mapping[a, h+1]] = 1 - self.alpha
            else:
                self.transitions[self.wait][state_index, 0] = 1
                self.reward_honest[self.wait][state_index, 0] = 10000

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
        return opt_policy
        
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

def main():
    for alpha in [0.35]:
        print(alpha)
        mdp = SelfishMDP(alpha=alpha, T=9, epsilon=10e-5)
        policy = mdp.getRhoBounds()
        mdp.processPolicy(policy)

if __name__ == "__main__":
    main()