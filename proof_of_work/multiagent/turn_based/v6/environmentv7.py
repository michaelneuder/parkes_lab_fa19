import numpy as np
np.random.seed(0)

IRRELEVANT = 0
RELEVANT = 1
ACTIVE = 2

class Environment(object):
    def __init__(self, alpha, gamma, T, mining_cost=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.current_state = None
        self.mining_cost = mining_cost

    def reset(self):
        self.current_state = (0, 0, IRRELEVANT)
        return self.current_state
    
    def getNextStateAdopt(self, rand_val):
        self.current_state = (0, 0, IRRELEVANT)
        return np.asarray(self.current_state), 0
    
    def getNextStateOverride(self, rand_val):
        a, h, _fork = self.current_state
        if a <= h:
            self.current_state = (0, 0, IRRELEVANT)
            return np.asarray(self.current_state), -100
        self.current_state = (a - h - 1, 0, IRRELEVANT)
        return np.asarray(self.current_state), h + 1
    
    def getNextStateMine(self, rand_val):
        a, h, fork = self.current_state
        if (a == self.T) or (h == self.T):
            return self.getNextStateAdopt(rand_val)
        if fork == ACTIVE:
            return self.backendMatchMine(rand_val)
        if rand_val < self.alpha:
            self.current_state = (a + 1, h, IRRELEVANT)
        else:
            self.current_state = (a, h + 1, RELEVANT)
        return np.asarray(self.current_state), -1*self.alpha*self.mining_cost
    
    def getNextStateMatch(self, rand_val):
        a, h, fork = self.current_state
        if (a < h) or (fork != RELEVANT):
            self.current_state = (0, 1, IRRELEVANT)
            return np.asarray(self.current_state), -100 
        return self.backendMatchMine(rand_val)
    
    def backendMatchMine(self, rand_val):
        a, h, _fork = self.current_state
        reward = -1*self.alpha*self.mining_cost
        if rand_val < self.alpha:
            self.current_state = (a + 1, h, ACTIVE)
        elif rand_val < self.gamma * (1 - self.alpha):
            self.current_state = (a - h, 1, RELEVANT)
            reward += h
        else: 
            self.current_state = (a, h+1, RELEVANT)
        return np.asarray(self.current_state), reward
    
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
        
def main():
    env = Environment(alpha=0.35, gamma=0.5, T=9)
    print(env.reset())
    print(env.takeAction(2, 0.01))
    print(env.takeAction(1, 0.01))

if __name__ == "__main__":
    main() 