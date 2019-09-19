from reduced_environment import Environment as e 
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

class QLearningAgent(object):
    def __init__(self, discount, alpha, T, rho):
        self.discount = discount
        self.env = None
        self.alpha = alpha
        self.T = T
        self.state_count = (T+1) * (T+1)
        self.qvalues = np.zeros((self.state_count, 3))
        self.rho = rho
        self.initial_learning_rate = 0.995
        self.min_learning_rate = 0.05
        self.initial_exploration_rate = 0.995
        self.min_exploration_rate = 0.1
        self.states_visited = np.zeros((self.T+1, self.T+1))

        # initialize state mapping and states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(T+1):
            for h in range(T+1):
                self.state_mapping[(a, h)] = count
                self.states.append((a, h))
                count += 1
    
    def chooseAction(self, current_state):
        legal_actions = self.env.getLegalActions()
        state_index = self.state_mapping[current_state]
        # explore
        current_explore_rate = np.power(self.initial_exploration_rate, self.states_visited[current_state])
        if current_explore_rate < self.min_exploration_rate:
            current_explore_rate = self.min_exploration_rate
        if np.random.uniform() < current_explore_rate:
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
        self.env = e(self.alpha, self.T)
        for _ in range(iterations):
            current_state_tuple = self.env.current_state.getTupleRepresentation()
            self.states_visited[current_state_tuple] += 1
            current_state_index = self.state_mapping[current_state_tuple]
            action = self.chooseAction(current_state_tuple)
            print('current_state: ', self.env.current_state, ' -- action: ', action)
            new_state, reward = self.env.takeAction(action)
            print('new_state: ', new_state, ' -- reward: ', reward)
            new_state_index = self.state_mapping[new_state.getTupleRepresentation()]
            reward_value = self.evalReward(reward)
            highest_qvalue_new_state = max(self.qvalues[new_state_index])
            
            current_learn_rate = np.power(self.initial_learning_rate, self.states_visited[current_state_tuple])
            if current_learn_rate < self.min_learning_rate:
                current_learn_rate = self.min_learning_rate
            # Q-values being updated
            sample = reward_value + self.discount*highest_qvalue_new_state
            self.qvalues[current_state_index, action] = (1 - current_learn_rate)*self.qvalues[current_state_index, action] + current_learn_rate*sample
    
    def runTrial_test(self, iterations):
        self.env = e(self.alpha, self.T)
        for _ in range(iterations):
            current_state_tuple = self.env.current_state.getTupleRepresentation()
            self.states_visited[current_state_tuple] += 1
            current_state_index = self.state_mapping[current_state_tuple]
            action = self.chooseAction(current_state_tuple)
            print('current_state: ', self.env.current_state, ' -- action: ', action)
            new_state, reward = self.env.takeAction(action)
            print('new_state: ', new_state, ' -- reward: ', reward)
            new_state_index = self.state_mapping[new_state.getTupleRepresentation()]
            reward_value = self.evalReward(reward)
            highest_qvalue_new_state = max(self.qvalues[new_state_index])
            
            current_learn_rate = 0.1
            # if current_learn_rate < self.min_learning_rate:
            #     current_learn_rate = self.min_learning_rate
            # Q-values being updated
            sample = reward_value + self.discount*highest_qvalue_new_state
            self.qvalues[current_state_index, action] = (1 - current_learn_rate)*self.qvalues[current_state_index, action] + current_learn_rate*sample
    
    def extractPolicy(self):
        policy = []
        for i in range(len(self.states)):
            a, h = self.states[i]
            # any action is legal
            if a > h:
                max_action = np.argmax(self.qvalues[i])
            else:
                arg_sorted = np.argsort(-self.qvalues[i])
                # override not legal
                if arg_sorted[0] == 1:
                    max_action = arg_sorted[1]
                else:
                    max_action = arg_sorted[0]
            policy.append(max_action)
        return policy

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
    
    def plotStatesVisited(self):
        f, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(self.states_visited, cmap='hot', interpolation='nearest')
        f.colorbar(im)
        plt.savefig('states_visited.png')
        plt.show()
    
    def plotLogStatesVisited(self):
        f, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(np.log(self.states_visited+1), cmap='hot', interpolation='nearest')
        f.colorbar(im)
        plt.savefig('log_states_visited.png')
        plt.show()

def main():
    qlagent = QLearningAgent(discount=1, alpha=1/3, T=9 , rho=0.33657073974609375)
    qlagent.runTrial(iterations=int(1000*10))
    qlagent.processPolicy(qlagent.extractPolicy())
    qlagent.plotStatesVisited()
    qlagent.plotLogStatesVisited()

if __name__ == "__main__":
    main()