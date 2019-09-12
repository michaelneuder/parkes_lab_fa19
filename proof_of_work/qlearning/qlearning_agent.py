from environment import Environment as e 
import matplotlib.pyplot as plt
import numpy as np

class QLearningAgent(object):
    def __init__(self, discount, alpha, T, gamma, rho):
        self.discount = discount
        self.env = e(alpha, T, gamma)
        self.state_count = (T+1) * (T+1) * 3
        self.action_count = 4
        self.qvalues = np.zeros((self.state_count, 4))
        self.rate = 1
        self.rho = rho
        
        # initialize state mapping and states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(T+1):
            for h in range(T+1):
                for fork in range(3):
                    self.state_mapping[(a, h, fork)] = count
                    self.states.append((a, h, fork))
                    count += 1
    
        
    def chooseAction(self, state_index):
        legal_actions = self.env.getLegalActions()
        # explore
        if np.random.uniform() < self.rate:
            return np.random.choice(legal_actions)
       
        # exploit
        current_action = -1
        current_value = float("-inf")
        for action in legal_actions:
            new_value = self.qvalues[state_index, action]
            if new_value > current_value:
                current_action = action
                current_value = new_value
        return current_action
    
    def evalReward(self, reward):
        return (1 - self.rho) * reward[0] - self.rho * reward[1]

    def runTrial(self, iterations):
        self.env.reset()
        for _ in range(iterations):
            current_state_tuple = self.env.current_state.getTupleRepresentation()
            current_state_index = self.state_mapping[current_state_tuple]
            action = self.chooseAction(current_state_index)
            print('current_state: ', self.env.current_state, ' -- action: ', action)
            new_state, reward = self.env.takeAction(action)
            print('new_state: ', new_state, ' -- reward: ', reward)
            new_state_index = self.state_mapping[new_state.getTupleRepresentation()]
            reward_value = self.evalReward(reward)
            highest_qvalue_new_state = max(self.qvalues[new_state_index])
            
            # Q-values being updated
            sample = reward_value + self.discount*highest_qvalue_new_state
            self.qvalues[current_state_index, action] = (1 - self.rate)*self.qvalues[current_state_index, action] + self.rate*sample

            # reduce exploration rate
            if self.rate > 0.01:
                self.rate *= 0.99
    
    def extractPolicy(self):
        policy = []
        for i in range(len(self.states)):
            max_action = np.argmax(self.qvalues[i])
            policy.append(max_action)
        return policy

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
                    results += 'w'
                elif action == 3:
                    results += 'm'
                else:
                    print('here')
                results += ' & '
            results += '\\\\ \n'
        print(results)
    
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
                        results += 'w'
                    elif action == 3:
                        results += 'm'
                    else:
                        print('here')
                results += ' & '
            results += '\\\\ \n'
        print(results)

def main():
    qlagent = QLearningAgent(discount=1, alpha=0.35, T=9, gamma=0, rho=0.36702728271484375)
    qlagent.runTrial(iterations=100000)
    qlagent.processPolicyIrrelevant(qlagent.extractPolicy())

if __name__ == "__main__":
    main()