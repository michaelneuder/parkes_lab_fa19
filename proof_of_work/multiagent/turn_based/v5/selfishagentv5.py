import numpy as np

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class SelfishAgent(object):
    def __init__(self, T):
        self.T = T
        self.policy = np.asarray([
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [2, 2, 2, 0, 0, 0, 0, 0, 0], 
            [2, 1, 2, 2, 2, 0, 0, 0, 0], 
            [2, 2, 1, 2, 2, 2, 0, 0, 0], 
            [2, 2, 2, 1, 2, 2, 2, 0, 0], 
            [2, 2, 2, 2, 1, 2, 2, 2, 0], 
            [2, 2, 2, 2, 2, 1, 2, 2, 0], 
            [2, 2, 2, 2, 2, 2, 1, 2, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

    def act(self, state):
        a, h = state
        if h == self.T:
            return ADOPT
        if a == self.T:
            return OVERRIDE
        if h > a:
            return ADOPT
        if (h == a-1) and (h >= 1):
            return OVERRIDE
        return WAIT
    
    def act2(self, state):
        action = self.policy[state]
        if action == 0:
            return 'adopt'
        if action == 1:
            return 'override'
        return 'wait'
    
    def step(self, state, action, new_state, reward):
        return

        