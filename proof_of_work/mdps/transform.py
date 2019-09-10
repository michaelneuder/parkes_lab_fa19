import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore', category=ss.SparseEfficiencyWarning)

class SelfishMDP(object):
    def __init__(self, alpha, T, gamma, epsilon): 
        self.alpha = alpha
        self.T = T
        self.gamma = gamma
        self.state_count = (T+1) * (T+1) * 3
        self.epsilon = epsilon
        self.initMDPHelpers()
        print('initializing matrices...')
        self.initMatrices()
        print('done!')
        print('writing to matrices...')
        self.writeMatrixData()
        print('done!')
    
    def initMDPHelpers(self):
        self.irrelevant = 0; self.relevant = 1; self.active = 2
        self.action_count = 4
        self.adopt = 0; self.override = 1; self.match = 2; self.wait = 3
        # generate a state to integer mapping and list of states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(self.T+1):
            for h in range(self.T+1):
                for fork in range(3):
                    self.state_mapping[(a, h, fork)] = count
                    self.states.append((a, h, fork))
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
            a, h, fork = self.states[state_index]
            
            # adopt transitions
            self.transitions[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = self.alpha
            self.transitions[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = 1 - self.alpha
            # adopt rewards
            self.reward_honest[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = h
            self.reward_honest[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = h

            # override
            if a > h:
                self.transitions[self.override][state_index, self.state_mapping[a-h, 0, self.irrelevant]] = self.alpha
                self.reward_selfish[self.override][state_index, self.state_mapping[a-h, 0, self.irrelevant]] = h+1
                self.transitions[self.override][state_index, self.state_mapping[a-h-1, 1, self.relevant]] = 1 - self.alpha
                self.reward_selfish[self.override][state_index, self.state_mapping[a-h-1, 1, self.relevant]] = h+1
            else:
                self.transitions[self.override][state_index, 0] = 1
                self.reward_honest[self.override][state_index, 0] = 10000

            # wait
            if (fork != self.active) and (a < self.T) and (h < self.T):
                self.transitions[self.wait][state_index, self.state_mapping[a+1, h, self.irrelevant]] = self.alpha
                self.transitions[self.wait][state_index, self.state_mapping[a, h+1, self.relevant]] = 1 - self.alpha
            elif (fork == self.active) and (a > h) and (h > 0) and (a < self.T) and (h < self.T): 
                self.transitions[self.wait][state_index, self.state_mapping[a+1, h, self.active]] = self.alpha
                self.transitions[self.wait][state_index, self.state_mapping[a-h, 1, self.relevant]] = self.gamma*(1-self.alpha)
                self.reward_selfish[self.wait][state_index, self.state_mapping[a-h, 1, self.relevant]] = h
                self.transitions[self.wait][state_index, self.state_mapping[a, h+1, self.relevant]] = (1-self.gamma)*(1-self.alpha)
            else:
                self.transitions[self.wait][state_index, 0] = 1
                self.reward_honest[self.wait][state_index, 0] = 10000

            # match
            if (fork == self.relevant) and (a >= h) and (h > 0) and (a < self.T) and (h < self.T):
                self.transitions[self.match][state_index, self.state_mapping[a+1, h, self.active]] = self.alpha
                self.transitions[self.match][state_index, self.state_mapping[a-h, 1, self.relevant]] = self.gamma*(1-self.alpha)
                self.reward_selfish[self.match][state_index, self.state_mapping[a-h, 1, self.relevant]] = h
                self.transitions[self.match][state_index, self.state_mapping[a, h+1, self.relevant]] = (1-self.gamma)*(1-self.alpha)
            else:
                self.transitions[self.match][state_index, 0] = 1
                self.reward_honest[self.match][state_index, 0] = 10000

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
        rho_prime = np.max(low - self.epsilon/4, 0)
        self.makeRewardsOverpay(rho_prime)
        total_reward_overpay = []
        for i in range(self.action_count):
            total_reward_overpay.append((1-rho_prime)*self.reward_selfish[i] - rho_prime*self.reward_honest[i])
        rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, total_reward_overpay, self.epsilon)
        rvi.run()
        upper_bound = rho_prime + 2 * (rvi.average_reward + self.epsilon)
        print('alpha: ', self.alpha, 'upper bound reward:', upper_bound)
        return opt_policy
        
    
    def makeRewardsOverpay(self, rho):
        for state_index in range(self.state_count):
            a, h, _fork = self.states[state_index]
            if a == self.T:
                expr = (1-rho)*self.alpha*(1-self.alpha)/(1-2*self.alpha)**2+0.5*((a-h)/(1-2*self.alpha)+a+h)
                self.reward_selfish[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = expr
                self.reward_selfish[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = expr
                self.reward_honest[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = 0
                self.reward_honest[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = 0
            elif h == self.T:
                expr1 = (1 - np.power(self.alpha/(1-self.alpha), h - a)) * (-1*rho*h)
                expr2 = np.power(self.alpha/(1-self.alpha), h - a) * (1 - rho)
                expr3 = (self.alpha * (1-self.alpha)) / (np.power(1-2*self.alpha, 2)) + (h - a) / (1- 2 * self.alpha)
                expr_total = expr1 + expr2 * expr3
                self.reward_selfish[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = expr_total
                self.reward_selfish[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = expr_total
                self.reward_honest[self.adopt][state_index, self.state_mapping[1, 0, self.irrelevant]] = 0
                self.reward_honest[self.adopt][state_index, self.state_mapping[0, 1, self.irrelevant]] = 0
    
    def processPolicy(self, policy):
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
                        results += 'm'
                    elif action == 3:
                        results += 'w'
                    else:
                        print('here')
                results += ' & '
            results += '\\\\ \n'
        print(results)

    def processPolicyIrrelevant(self, policy):
        results = ''
        for a in range(9):
            for h in range(9):
                state_index = self.state_mapping[(a, h, 0)]
                action = policy[state_index]
                if action == 0:
                    results += 'a'
                elif action == 1:
                    results += 'o'
                elif action == 2:
                    results += 'm'
                elif action == 3:
                    results += 'w'
                else:
                    print('here')
                results += ' & '
            results += '\\\\ \n'
        print(results)


def main():
    for alpha in [0.35]:
        print(alpha)
        mdp = SelfishMDP(alpha=alpha, T=9, gamma=0.0, epsilon=10e-5)
        policy = mdp.getRhoBounds()
        mdp.processPolicyIrrelevant(policy)

if __name__ == "__main__":
    main()