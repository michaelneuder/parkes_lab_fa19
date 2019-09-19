from collections import deque
import copy
import deep_q_env
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import random
np.random.seed(0)

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size, T):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.rho = 0.36702728271484375
        self.current_model = self._build_model()
        self.target_model = self._build_model()

        # initialize state mapping and states
        self.state_mapping = {}
        self.states = []
        count = 0
        for a in range(T+1):
            for h in range(T+1):
                self.state_mapping[(a, h)] = count
                self.states.append((a, h))
                count += 1
        

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def syncModels(self):
        self.target_model = copy.deepcopy(self.current_model)
    
    def evalReward(self, reward):
        return (1 - self.rho) * reward[0] - self.rho  * reward[1]

    #get action
    def act(self, state):
        #select random action with prob=epsilon else action=maxQ
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.current_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        #sample random transitions
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            reward_eval = self.evalReward(reward)
            target = reward_eval
            if not done:
                Q_next=self.target_model.predict(next_state)[0]
                target = (reward_eval + self.gamma * np.amax(Q_next))

            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            #train network
            self.current_model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

env = deep_q_env.Environment(alpha=1/3, T=9)
state_size = 2
action_size = 3
agent = DQNAgent(state_size, action_size, T=9)
batch_size = 32

bar = progressbar.ProgressBar()
for e in bar(range(100)):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.takeAction(action)
        next_state = np.reshape(next_state, [1, state_size])
        #add to experience memory
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        #experience replay
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.syncModels()

policy = agent.extractPolicy()
agent.processPolicy(policy)