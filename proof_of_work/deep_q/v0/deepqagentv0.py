import copy
from environmentv0 import Environment
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import util
np.random.seed(0)

class DeepQLearningAgent(object):
    def __init__(self, discount, alpha, T, rho):
        # MDP
        self.alpha = alpha
        self.T = T
        self.rho = rho
        self.exploration_rate = 1
        self.exploration_decrease = 0.001
        self.min_exploration_rate = 0.1
        
        # deep q
        self.learning_rate = 0.001
        self.value_model = util.createModel(self.learning_rate)
        self.target_model = None
        self.learning_update_count = 0
        self.memories = []
        self.training_memory_count = 32
        self.discount = discount
        self.update_target_frequency = 100
        self.max_memory_count = 3000
        
        # environment
        self.env = Environment(self.alpha, self.T)
        
        # visualization
        self.states_visited = np.zeros((self.T+1, self.T+1))
    
    def chooseAction(self, current_state):
        # finding rate of exploration. Linear decay to minimum.
        if self.exploration_rate < self.min_exploration_rate:
            current_explore_rate = self.min_exploration_rate
        else:
            self.exploration_rate -= self.exploration_decrease
            current_explore_rate = self.exploration_rate

        if np.random.uniform() < current_explore_rate:
            return np.random.choice([0,1,2])
        return np.argmax(self.value_model.predict(util.prepareInput(current_state)))
    
    def syncModels(self):
        self.target_model = copy.deepcopy(self.value_model)

    def learn(self, iterations=10000):
        bar = progressbar.ProgressBar()
        for _ in bar(range(iterations)):
            self.runTrial()

    def runTrial(self):
        done = False
        self.env.reset()
        while not done:
            current_state = self.env.current_state
            self.states_visited[current_state] += 1
            
            # take action
            action = self.chooseAction(current_state)
            new_state, reward, done = self.env.takeAction(action)
            reward_value = util.evalReward(self.rho, reward)

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
            if len(self.memories) > self.max_memory_count:
                self.memories.pop(0)
            if self.learning_update_count % self.update_target_frequency == 0:
                self.syncModels()
    
    def trainNeuralNet(self):
        memory_subset = np.random.choice(self.memories, self.training_memory_count, replace=False)
        training_data, training_target = [], []
        for memory in memory_subset:
            total_reward = memory['reward']
            
            # target net used to predict value of new state
            if not memory['done']:
                total_reward += self.discount * max(self.target_model.predict(util.prepareInput(memory['new_state']))[0])
            target = self.value_model.predict(util.prepareInput(memory['current_state']))
            
            # this modifies the prediction to have new value for the action taken
            target[0][memory['action']] = total_reward
            training_data.append(memory['current_state'])
            training_target.append(target)

        # fiting model --- this is the neural net training 
        self.value_model.fit(
            np.squeeze(np.asarray(training_data)), 
            np.squeeze(np.asarray(training_target)), 
            epochs=1, 
            verbose=False)

def main():
    qlagent = DeepQLearningAgent(discount=1, alpha=1/3, T=9 , rho=0.33657073974609375)
    qlagent.learn(iterations=int(6))
    
    # results
    analyzer = util.ResultsAnalyzer(qlagent.value_model, qlagent.states_visited)
    end_policy = analyzer.extractPolicy()
    analyzer.processPolicy(end_policy)
    analyzer.plotStatesVisited()
    analyzer.plotLogStatesVisited()

if __name__ == "__main__":
    main()