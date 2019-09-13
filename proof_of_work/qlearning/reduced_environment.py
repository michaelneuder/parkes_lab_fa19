import numpy as np
np.random.seed(0)

ADOPT = 0
OVERRIDE = 1
WAIT = 2

class State(object):
    def __init__(self, a, h):
        self.a = a
        self.h = h
    
    def __str__(self):
        return 'a={}, h={}'.format(self.a, self.h)
    
    def getTupleRepresentation(self):
        return (self.a, self.h)

class Environment(object):
    def __init__(self, alpha, T, rand_val=None):
        self.alpha = alpha
        self.T = T
        self.current_state = self.getInitialState(rand_val)

    def getInitialState(self, rand_val):
        if not rand_val:
            rand_val = np.random.uniform()
        if rand_val < self.alpha:
            return State(1, 0)
        return State(0, 1)
    
    def getNextStateAdopt(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(1, 0)
            reward = (0, self.current_state.h)
        else:
            new_state = State(0, 1)
            reward = (0, self.current_state.h)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateOverride(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a - self.current_state.h, 0)
        else:
            new_state = State(self.current_state.a - self.current_state.h - 1, 1)
        reward = (self.current_state.h+1, 0)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateWait(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a + 1, self.current_state.h)
        else:
            new_state = State(self.current_state.a, self.current_state.h + 1)
        self.current_state = new_state
        return new_state, (0, 0)
        
    def getLegalActions(self):
        assert(self.current_state.a <= self.T)
        assert(self.current_state.h <= self.T)
        actions = [ADOPT]
        if (self.current_state.a == self.T) or (self.current_state.h == self.T):
            return actions
        actions.append(WAIT)
        if self.current_state.a > self.current_state.h:
            actions.append(OVERRIDE)
        return actions
    
    def takeAction(self, action, rand_val=None):
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