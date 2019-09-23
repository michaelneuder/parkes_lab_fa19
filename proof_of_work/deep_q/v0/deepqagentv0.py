import copy
from environmentv0 import Environment
from keras.models import clone_model
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
        self.exploration_decrease = 0.0001
        self.min_exploration_rate = 0.1
        
        # deep q
        self.learning_rate = 0.001
        self.value_model = util.createModel(self.learning_rate)
        self.target_model = copy.deepcopy(self.value_model)
        self.learning_update_count = 0
        self.memories = []
        self.training_memory_count = 32
        self.discount = discount
        self.update_target_frequency = 1000
        self.max_memory_count = 10000
        self.min_memory_count_learn = 1000
        
        # environment
        self.env = Environment(self.alpha, self.T)
        
        # visualization
        self.states_visited = np.zeros((self.T+1, self.T+1))
        self.steps_before_done = []
    
    def chooseAction(self, current_state):
        # explore based on number of visits to that state.
        times_visited = self.states_visited[current_state]
        current_explore_rate = 1 - times_visited * self.exploration_decrease
        if current_explore_rate < self.min_exploration_rate:
            current_explore_rate = self.min_exploration_rate
        
        if np.random.uniform() < current_explore_rate:
            return np.random.randint(low=0, high=3)
        return np.argmax(self.value_model.predict(util.prepareInput(current_state)))
    
    def syncModels(self):
        self.target_model = clone_model(self.value_model)
        self.target_model.set_weights(self.value_model.get_weights())

    def learn(self, iterations=10000):
        bar = progressbar.ProgressBar()
        for _ in bar(range(iterations)):
            self.runTrial()

    def runTrial(self):
        done = False
        self.env.reset()
        step_counter = 0
        while not done:
            step_counter += 1
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
            if len(self.memories) > self.min_memory_count_learn:
                self.trainNeuralNet()
                self.learning_update_count += 1

                # keep memory list finite
                if len(self.memories) > self.max_memory_count:
                    self.memories.pop(0)
                
                # update models
                if self.learning_update_count % self.update_target_frequency == 0:
                    print('global step: {}. syncing models'.format(self.learning_update_count))
                    self.syncModels()
                    self.value_model.save('saved_models/value_net_iter{0:06d}.h5'.format(self.learning_update_count))
        self.steps_before_done.append(step_counter)
    
    def trainNeuralNet(self):
        memory_subset_indeces = np.random.randint(low=0, high=len(self.memories), size=self.training_memory_count)
        memory_subset = [self.memories[i] for i in memory_subset_indeces]
        rewards = []
        current_states = []
        new_states = []
        actions = []
        dones = []
        for memory in memory_subset:
            rewards.append(memory['reward'])
            current_states.append(memory['current_state'])
            new_states.append(memory['new_state'])
            actions.append(memory['action'])
            dones.append(memory['done'])
            
        # current_state_predictions = self.value_model.predict(util.prepareInputs(current_states))
        current_state_predictions = np.zeros((len(current_states), 3))
        new_state_predictions = self.target_model.predict(util.prepareInputs(new_states))

        for i in range(len(new_state_predictions)):
            total_reward = rewards[i]
            if not dones[i]:
                total_reward += self.discount * max(new_state_predictions[i])
            
            # clip
            if total_reward > 1:
                total_reward = 1
            elif total_reward < -1:
                total_reward = -1
            
            current_state_predictions[i][actions[i]] = total_reward

        # fiting model --- this is the neural net training 
        self.value_model.fit(
            np.squeeze(np.asarray(current_states)), 
            np.squeeze(np.asarray(current_state_predictions)), 
            epochs=1, 
            verbose=False)

def main():
    qlagent = DeepQLearningAgent(discount=0.99, alpha=0.45, T=9 , rho=0.6032638549804688)
    qlagent.learn(iterations=int(20000))
    
    # results
    analyzer = util.ResultsAnalyzer(qlagent.value_model, qlagent.states_visited, qlagent.steps_before_done)
    end_policy = analyzer.extractPolicy()
    analyzer.processPolicy(end_policy)
    analyzer.plotStatesVisited(save=True)
    analyzer.plotLogStatesVisited(save=True)
    analyzer.plotStepsCounter(save=True)
    analyzer.plotExploration(save=True)

if __name__ == "__main__":
    main()