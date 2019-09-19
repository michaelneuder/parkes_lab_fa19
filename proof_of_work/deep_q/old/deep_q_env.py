import numpy as np
np.random.seed(0)

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class Environment(object):
    def __init__(self, alpha, T, rand_val=None):
        self.alpha = alpha
        self.T = T
        self.current_state = None

    def reset(self, rand_val=None):
        if not rand_val:
            rand_val = np.random.uniform()
        if rand_val < self.alpha:
            self.current_state = (1, 0)
        self.current_state = (0, 1)
        return self.current_state
    
    def getNextStateAdopt(self, rand_val):
        if rand_val < self.alpha:
            new_state = (1, 0)
        else:
            new_state = (0, 1)
        self.current_state = new_state
        return new_state, (0, self.current_state[1]), False
    
    def getNextStateOverride(self, rand_val):
        if self.current_state[0] <= self.current_state[1]:
            return (0,1), (0, 10000), True
        if rand_val < self.alpha:
            new_state = (self.current_state[0] - self.current_state[1], 0)
        else:
            new_state = (self.current_state[0] - self.current_state[1] - 1, 1)
        reward = (self.current_state[1]+1, 0)
        self.current_state = new_state
        return new_state, reward, False
    
    def getNextStateWait(self, rand_val):
        if rand_val < self.alpha:
            new_state = (self.current_state[0] + 1, self.current_state[1])
        else:
            new_state = (self.current_state[0], self.current_state[1] + 1)
        self.current_state = new_state
        return new_state, (0, 0), False
    
    def takeAction(self, action, rand_val=None):
        if (self.current_state[0] == self.T) or (self.current_state[1] == self.T):
            return (0, 1), (0, 10000), True
        if not rand_val:
            rand_val = np.random.uniform()
        assert(action in [0, 1, 2])
        if action == 0:
            return self.getNextStateAdopt(rand_val)
        elif action == 1:
            return self.getNextStateOverride(rand_val)
        else:
            return self.getNextStateWait(rand_val)
        
def main():
    env = Environment(alpha=0.35, T=9)
    print(env.current_state)

if __name__ == "__main__":
    main() 