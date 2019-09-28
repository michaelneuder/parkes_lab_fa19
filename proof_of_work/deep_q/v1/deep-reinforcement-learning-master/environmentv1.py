import numpy as np
np.random.seed(0)

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class Environment(object):
    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T
        self.current_state = None

    def reset(self, rand_val=None):
        # resents env to one of original states and returns the state.
        if not rand_val:
            rand_val = np.random.uniform()
        if rand_val < self.alpha:
            self.current_state = (1, 0)
        else:
            self.current_state = (0, 1)
        return self.current_state
    
    def getNextStateAdopt(self, rand_val):
        _a, h = self.current_state
        if rand_val < self.alpha:
            self.current_state = (1, 0)
        else:
            self.current_state = (0, 1)
        return np.asarray(self.current_state), (0, h)
    
    def getNextStateOverride(self, rand_val):
        a, h = self.current_state
        if a <= h:
            self.current_state = (0, 1)
            return np.asarray(self.current_state), (0, 10)
        if rand_val < self.alpha:
            self.current_state = (a - h, 0)
        else:
            self.current_state = (a - h - 1, 1)
        reward = (h+1, 0)
        return np.asarray(self.current_state), reward
    
    def getNextStateWait(self, rand_val):
        a, h = self.current_state
        if (a+1 == self.T) or (h+1 == self.T):
            return self.getNextStateAdopt(rand_val)
        if rand_val < self.alpha:
            self.current_state = (a + 1, h)
        else:
            self.current_state = (a, h + 1)
        return np.asarray(self.current_state), (0, 0)
    
    def takeAction(self, action, rand_val=None):
        assert(action in [0, 1, 2])
        if not rand_val:
            rand_val = np.random.uniform()
        if action == 0:
            return self.getNextStateAdopt(rand_val)
        elif action == 1:
            return self.getNextStateOverride(rand_val)
        else:
            return self.getNextStateWait(rand_val)
        
def main():
    env = Environment(alpha=0.35, T=9)
    print(env.reset(0.01))
    print(env.takeAction(2, 0.01))
    print(env.takeAction(1, 0.01))

if __name__ == "__main__":
    main() 