import numpy as np

ADOPT = 0
OVERRIDE = 1
MINE = 2
MATCH = 3

class SelfishAgent(object):
    def __init__(self, T):
        self.T = T

    def act(self, state):
        a, h, fork = state
        if h == self.T:
            return ADOPT
        if a == self.T:
            return OVERRIDE
        if h > a:
            return ADOPT
        if (h == 1) and (a == 1) and (fork == 2):
            return MINE
        if (h == 1) and (a == 1):
            return MATCH
        if (h == a-1) and (h >= 1):
            return OVERRIDE
        return MINE
    
    def step(self, state, action, new_state, reward):
        return

        