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
        self.T = T
        
    def extractPolicy(self):
        policy = np.zeros((self.T, self.T)) - 1
        for a in range(self.T):
            for h in range(self.T):
                max_action = np.argmax(self.value_model.predict(prepareInput((a,h)))[0])
                policy[a,h] = max_action
        return policy

    def processPolicy(self, policy):
        results = ''
        print(policy)
        for a in range(9):
            results += '{} & '.format(a)
            for h in range(9):
                action = policy[a, h]
                assert(action in [0, 1, 2])
                if action == 0:
                    results += '\\ag'
                elif action == 1:
                    results += '\\ob'
                else:
                    results += '\\wt'
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
        # plt.show()
    
    def plotLogStatesVisited(self, save=False):
        f, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(np.log(self.states_visited+1), cmap='hot', interpolation='nearest')
        f.colorbar(im)
        if save:
            plt.savefig('img/log_states_visited.png')
        # plt.show()
    
    def plotStepsCounter(self, save=False):
        _f, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel('steps taken before termination')
        ax.set_xlabel('trial number')
        ax.plot(self.steps_per_trial)
        ravgs = [sum(self.steps_per_trial[i:i+50])/50. for i in range(len(self.steps_per_trial)-49)]
        ax.plot(ravgs, 'r--')
        if save:
            plt.savefig('img/steps_per_trials.png')
        # plt.show()
    
    def plotExploration(self, save=False):
        f, ax = plt.subplots(figsize=(10,10))
        exploration = 1 - self.states_visited*0.01
        exploration[exploration < 0.1] = 0.1
        im = ax.imshow(exploration, cmap='hot', interpolation='nearest')
        f.colorbar(im)
        if save:
            plt.savefig('img/exploration_rate.png')
        # plt.show()