import numpy as np

IRRELEVANT = 0
RELEVANT = 1
ACTIVE = 2

class Environment(object):
    def __init__(self, alpha, gamma, T, mining_cost):
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.mining_cost = mining_cost
        self.current_state = None

    def reset(self):
        self.current_state = (0, 0, IRRELEVANT)
        return self.current_state
    
    def getNextStateAdopt(self, rand_val):
        self.current_state = (0, 0, IRRELEVANT)
        return np.asarray(self.current_state), 0
    
    def getNextStateOverride(self, rand_val):
        a, h, fork = self.current_state
        if a <= h:
            raise RuntimeError('Illegal Override, state={}'.format((a, h, fork)))
        self.current_state = (a - h - 1, 0, IRRELEVANT)
        return np.asarray(self.current_state), h + 1
    
    def getNextStateMine(self, rand_val):
        a, h, fork = self.current_state
        if (fork != ACTIVE) and (a < self.T) and (h < self.T):
            if rand_val < self.alpha:
                self.current_state = (a + 1, h, IRRELEVANT)
            else:
                self.current_state = (a, h + 1, RELEVANT)
            return np.asarray(self.current_state), -1*self.alpha*self.mining_cost
        elif (fork == ACTIVE) and (a > h) and (h > 0) and (a < self.T) and (h < self.T):
            return self.backendMatchMine()
        else:
            raise RuntimeError('Illegal Mine, state={}'.format((a, h, fork)))
    
    def getNextStateMatch(self, rand_val):
        a, h, fork = self.current_state
        if (fork == RELEVANT) and (a >= h) and (h > 0) and (a < self.T) and (h < self.T):
            return self.backendMatchMine()
        raise RuntimeError('Illegal Match, state={}'.format((a, h, fork)))
    
    def backendMatchMine(self):
        a, h, _fork = self.current_state
        outcome = np.random.choice([0, 1, 2], 
                                   p=[self.alpha, self.gamma*(1 - self.alpha), (1 - self.gamma) * (1 - self.alpha)])
        if outcome == 0:
            self.current_state = (a + 1, h, ACTIVE)
            return np.asarray(self.current_state), -1*self.alpha*self.mining_cost
        elif outcome == 1:
            self.current_state = (a - h, 1, RELEVANT)
            return np.asarray(self.current_state), h-1*self.alpha*self.mining_cost
        else: 
            self.current_state = (a, h+1, RELEVANT)
            return np.asarray(self.current_state), -1*self.alpha*self.mining_cost
    
    def takeAction(self, action, rand_val=None):
        assert(action in [0, 1, 2, 3])
        if not rand_val:
            rand_val = np.random.uniform()
        if action == 0:
            return self.getNextStateAdopt(rand_val)
        elif action == 1:
            return self.getNextStateOverride(rand_val)
        elif action == 2:
            return self.getNextStateMine(rand_val)
        else:
            return self.getNextStateMatch(rand_val)
