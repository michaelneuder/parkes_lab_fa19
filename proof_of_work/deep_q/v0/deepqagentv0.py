import copy
from environmentv0 import Environment
from keras.models import clone_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import time
import util
np.random.seed(0)

class DeepQLearningAgent(object):
    def __init__(self, discount, alpha, T, rho):
        # MDP
        self.alpha = alpha
        self.T = T
        self.rho = rho
        self.exploration_rate = 1
        self.exploration_decrease = float(1e-5)
        self.min_exploration_rate = 0.1
        
        # deep q
        self.learning_rate = 0.001
        self.value_model = util.createModel(self.learning_rate)
        self.target_model = clone_model(self.value_model)
        self.target_model.set_weights(self.value_model.get_weights())
        self.learning_update_count = 0
        self.max_learning_steps = int(2e5)
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
        self.last_50_steps = []
        self.snyc_points = []

        # timing
        self.last_target_net_clone = time.time()
    
    def chooseAction(self, current_state):
        # explore based on number of visits to that state.
        self.exploration_rate -= self.exploration_decrease
        current_explore_rate = self.exploration_rate
        if self.exploration_rate < self.min_exploration_rate:
            current_explore_rate = self.min_exploration_rate
        
        if np.random.uniform() < current_explore_rate:
            return np.random.randint(low=0, high=3)
        return np.argmax(self.value_model.predict(util.prepareInput(current_state)))
    
    def syncModels(self):
        self.target_model = clone_model(self.value_model)
        self.target_model.set_weights(self.value_model.get_weights())

    def learn(self, iterations=10000):
        start_time = time.time()
        while self.learning_update_count < self.max_learning_steps:
            self.runTrial()
        print("total time {:.04f} s".format(time.time() - start_time))

    def runTrial(self):
        done = False
        self.env.reset()
        step_counter = 0
        while (not done) and (self.learning_update_count < self.max_learning_steps):
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
                    update_time = time.time() - self.last_target_net_clone
                    print('    last synced: {:.04f} s ago'.format(update_time))
                    updates_remaining = (self.max_learning_steps - self.learning_update_count)/ self.update_target_frequency
                    print('    eta: {:.02f} s'.format(updates_remaining * update_time))
                    print('*'*30)
                    self.syncModels()
                    self.value_model.save('saved_models/value_net_iter{0:06d}.h5'.format(self.learning_update_count))
                    self.snyc_points.append(self.learning_update_count)
                    self.last_50_steps.append(np.mean(self.steps_before_done[-50:]))
                    self.last_target_net_clone = time.time()
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
    qlagent = DeepQLearningAgent(discount=0.99, alpha=1/3, T=9 , rho=0.33657073974609375)
    qlagent.learn(iterations=int(5000))
    print(qlagent.exploration_rate)
    
    # results
    analyzer = util.ResultsAnalyzer(
        qlagent.value_model, qlagent.states_visited, qlagent.steps_before_done, 
        qlagent.last_50_steps, qlagent.snyc_points)
    end_policy = analyzer.extractPolicy()
    analyzer.processPolicy(end_policy)
    analyzer.plotStatesVisited(save=True)
    analyzer.plotLogStatesVisited(save=True)
    analyzer.plotStepsCounter(save=True)
    analyzer.plotExploration(save=True)
    analyzer.plotLast50(save=True)

if __name__ == "__main__":
    main()