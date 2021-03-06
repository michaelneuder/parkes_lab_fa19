import numpy as np
np.random.seed(0)

ADOPT = 0
OVERRIDE = 1
WAIT = 2
TERMINAL_STATE = (0,0)

class Environment(object):
    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T
        self.current_state = TERMINAL_STATE

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
        self.reset()
        # reward = (0, h)
        return np.asarray(self.current_state), (0, self.current_state[1]), True
    
    def getNextStateOverride(self, rand_val):
        if self.current_state[0] <= self.current_state[1]:
            self.current_state = TERMINAL_STATE
            return np.asarray(TERMINAL_STATE), (0, 10), True
        # new state = (a-h, 0)
        if rand_val < self.alpha:
            new_state = (self.current_state[0] - self.current_state[1], 0)
        # new state = (a-h-1, 1)
        else:
            new_state = (self.current_state[0] - self.current_state[1] - 1, 1)
        # reward = (h+1, 0)
        reward = (self.current_state[1]+1, 0)
        self.current_state = new_state
        return np.asarray(new_state), reward, True
    
    def getNextStateWait(self, rand_val):
        # new state = (a+1, h)
        if rand_val < self.alpha:
            new_state = (self.current_state[0] + 1, self.current_state[1])
        # new state = (a, h+1)
        else:
            new_state = (self.current_state[0], self.current_state[1] + 1)
        # check terminal state
        if (new_state[0] == self.T) or (new_state[1] == self.T):
            self.current_state = TERMINAL_STATE
            # reward = (0, h)
            return np.asarray(TERMINAL_STATE), (0, self.current_state[1]), True
        self.current_state = new_state
        return np.asarray(new_state), (0, 0), False
    
    def takeAction(self, action, rand_val=None):
        assert self.current_state != TERMINAL_STATE
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