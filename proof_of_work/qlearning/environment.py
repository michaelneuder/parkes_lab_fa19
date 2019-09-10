import numpy as np

class State(object):
    def __init__(self, a, h, fork):
        self.a = a
        self.h = h
        self.fork = fork
    
    def __str__(self):
        return 'a={}, h={}, fork={}'.format(self.a, self.h, self.fork)

class Environment(object):
    def __init__(self, alpha, T, gamma):
        self.alpha = alpha
        self.T = T
        self.gamma = gamma
        self.irrelevant = 0; self.relevant = 1; self.active = 2
        self.adopt = 0; self.override = 1; self.match = 2; self.wait = 3
        self.current_state = self.getInitialState()
        
    def getInitialState(self, rand_val):
        if rand_val < self.alpha:
            return State(1, 0, self.irrelevant)
        return State(0, 1, self.irrelevant)
    
    def getNextStateAdopt(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(1, 0, self.irrelevant)
            reward = (0, self.current_state.h)
        else:
            new_state = State(0, 1, self.irrelevant)
            reward = (0, self.current_state.h)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateOverride(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a - self.current_state.h, 0 , self.irrelevant)
        else:
            new_state = State(self.current_state.a - self.current_state.h - 1, 1 , self.relevant)
        reward = (self.current_state.h+1, 0)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateWaitInactive(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a + 1, self.current_state.h, self.irrelevant)
        else:
            new_state = State(self.current_state.a, self.current_state.h + 1, self.relevant)
        self.current_state = new_state
        return new_state, (0, 0)
    
    def getNextStateWaitMatch(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a + 1, self.current_state.h, self.active)
            reward = (0, 0)
        elif rand_val < self.gamma * (1 - self.alpha):
            new_state = State(self.current_state.a -self.current_state.h,  1, self.relevant)
            reward = (h, 0)
        else:
            new_state = State(self.current_state.a, self.current_state.h + 1, self.relevant)
            reward = (0, 0)
        self.current_state = new_state
        return new_state, (0, 0)
        
        





def main():
    env = Environment(alpha=0.35, T=9, gamma=0)
    print(env.current_state)

if __name__ == "__main__":
    main() 