from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def evalReward(rho, reward):
    return (1 - rho) * reward[0] - rho * reward[1]

def createModel(learning_rate):
    model = Sequential()
    model.add(Dense(12, input_dim=2, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

def prepareInput(state):
    return np.reshape(np.asarray(state), (1,2))

def prepareInputs(states):
    return np.asarray(states)

class ResultsAnalyzer(object):
    def __init__(self, model, states_visited, steps, T=9):
        self.value_model = model
        self.states_visited = states_visited
        self.steps_per_trial = steps
        
        # initialize state mapping and states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(T+1):
            for h in range(T+1):
                self.state_mapping[(a, h)] = count
                self.states.append((a, h))
                count += 1
    
    def extractPolicy(self):
        policy = []
        for state in self.states:
            a, h = state
            # any action is legal
            if a > h:
                max_action = np.argmax(self.value_model.predict(prepareInput(state))[0])
            else:
                arg_sorted = np.argsort(self.value_model.predict(prepareInput(state))[0])
                # override not legal
                if arg_sorted[0] == 1:
                    max_action = arg_sorted[1]
                else:
                    max_action = arg_sorted[0]
            policy.append(max_action)
        return policy

    def processPolicy(self, policy):
        results = ''
        print(policy)
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
    
    def plotStatesVisited(self, save=False):
        f, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(self.states_visited, cmap='hot', interpolation='nearest')
        f.colorbar(im)
        if save:
            plt.savefig('img/states_visited.png')
        plt.show()
    
    def plotLogStatesVisited(self, save=False):
        f, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(np.log(self.states_visited+1), cmap='hot', interpolation='nearest')
        f.colorbar(im)
        if save:
            plt.savefig('img/log_states_visited.png')
        plt.show()
    
    def plotStepsCounter(self, save=False):
        f, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel('steps taken before termination')
        ax.set_xlabel('trial number')
        ax.plot(self.steps_per_trial)
        ravgs = [sum(self.steps_per_trial[i:i+10])/10. for i in range(len(self.steps_per_trial)-9)]
        ax.plot(ravgs, 'g--')
        if save:
            plt.savefig('img/steps_per_trials.png')
        plt.show()