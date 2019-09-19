import copy
from deep_q_env import Environment as e 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import progressbar
np.random.seed(0)

class DeepQLearningAgent(object):
    def __init__(self, discount, alpha, T, rho):
        self.discount = discount
        self.env = None
        self.alpha = alpha
        self.T = T
        self.rho = rho
        self.learning_rate = 0.001 
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
        
        # deep q
        self.current_model = self.initializeModel()
        self.target_model = copy.deepcopy(self.current_model)
        self.memories = []
        self.training_memory_count = 5
    
    def initializeModel(self):
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def chooseAction(self, current_state):
        current_explore_rate = np.power(self.initial_exploration_rate, self.states_visited[current_state])
        if current_explore_rate < self.min_exploration_rate:
            current_explore_rate = self.min_exploration_rate
        if np.random.uniform() < current_explore_rate:
            return np.random.randint(0, 3)
        
        act_values = self.current_model.predict(self.prepareInput(current_state))
        return np.argmax(act_values[0])
    
    def evalReward(self, reward):
        return (1 - self.rho) * reward[0] - self.rho * reward[1]

    def syncModels(self):
        self.target_model = copy.deepcopy(self.current_model)

    def learn(self, iterations):
        self.env = e(self.alpha, self.T)
        self.initializeModel()
        bar = progressbar.ProgressBar()
        for _ in bar(range(iterations)):
            self.env.reset()
            self.runTrial()
            self.syncModels()
        
    def runTrial(self):
        done = False
        while not done:
            current_state = self.env.current_state
            self.states_visited[current_state] += 1
            action = self.chooseAction(current_state)
            new_state, reward, done = self.env.takeAction(action)
            reward_value = self.evalReward(reward)
            
            # creating a new memory
            memory = dict({
                'current_state' : current_state,
                'action' : action,
                'reward' : reward_value,
                'new_state' : new_state,
                'done' : done
            })
            self.memories.append(memory)

            # training network
            if len(self.memories) > self.training_memory_count:
                self.trainNeuralNet()
            if len(self.memories) > 3000:
                self.memories.pop(0)
        print(self.memories)
    
    def trainNeuralNet(self):
        memory_subset = np.random.choice(self.memories, self.training_memory_count, replace=False)
        training_data, training_target = [], []
        for memory in memory_subset:
            total_reward = memory['reward']
            if memory['done']:
                total_reward += self.discount * max(self.target_model.predict(self.prepareInput(memory['new_state']))[0])
            target = self.target_model.predict(self.prepareInput(memory['current_state']))

            # this modifies the prediction to have new value for the action taken
            target[0][memory['action']] = total_reward
            training_data.append(memory['current_state'])
            training_target.append(target)

        # fiting model --- this is the neural net training 
        self.current_model.fit(
            np.squeeze(np.asarray(training_data)), 
            np.squeeze(np.asarray(training_target)), 
            epochs=1, 
            verbose=False)
    
    def prepareInput(self, state):
        return np.reshape(np.asarray(state), (1,2))
            
    def extractPolicy(self):
        policy = []
        for state in self.states:
            a, h = state
            # any action is legal
            if a > h:
                max_action = np.argmax(self.current_model.predict(self.prepareInput(state))[0])
            else:
                arg_sorted = np.argsort(self.current_model.predict(self.prepareInput(state))[0])
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
    qlagent = DeepQLearningAgent(discount=1, alpha=1/3, T=9 , rho=0.33657073974609375)
    qlagent.learn(iterations=int(10))
    qlagent.processPolicy(qlagent.extractPolicy())
    qlagent.plotStatesVisited()
    qlagent.plotLogStatesVisited()

if __name__ == "__main__":
    main()