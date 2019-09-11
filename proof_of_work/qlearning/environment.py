import numpy as np

IRRELEVANT = 0
RELEVANT = 1
ACTIVE = 2
ADOPT = 0
OVERRIDE = 1
WAIT = 2
MATCH = 3

class State(object):
    def __init__(self, a, h, fork):
        self.a = a
        self.h = h
        self.fork = fork
    
    def __str__(self):
        return 'a={}, h={}, fork={}'.format(self.a, self.h, self.fork)


class Environment(object):
    def __init__(self, alpha, T, gamma, rand_val=np.random.uniform()):
        self.alpha = alpha
        self.T = T
        self.gamma = gamma
        self.current_state = self.getInitialState(rand_val)
        
    def getInitialState(self, rand_val):
        if rand_val < self.alpha:
            return State(1, 0, IRRELEVANT)
        return State(0, 1, IRRELEVANT)
    
    def getNextStateAdopt(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(1, 0, IRRELEVANT)
            reward = (0, self.current_state.h)
        else:
            new_state = State(0, 1, IRRELEVANT)
            reward = (0, self.current_state.h)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateOverride(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a - self.current_state.h, 0 , IRRELEVANT)
        else:
            new_state = State(self.current_state.a - self.current_state.h - 1, 1 , RELEVANT)
        reward = (self.current_state.h+1, 0)
        self.current_state = new_state
        return new_state, reward
    
    def getNextStateWaitInactive(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a + 1, self.current_state.h, IRRELEVANT)
        else:
            new_state = State(self.current_state.a, self.current_state.h + 1, RELEVANT)
        self.current_state = new_state
        return new_state, (0, 0)
    
    def getNextStateWaitMatch(self, rand_val):
        if rand_val < self.alpha:
            new_state = State(self.current_state.a + 1, self.current_state.h, ACTIVE)
            reward = (0, 0)
        elif rand_val < self.gamma * (1 - self.alpha):
            new_state = State(self.current_state.a -self.current_state.h,  1, RELEVANT)
            reward = (self.current_state.h, 0)
        else:
            new_state = State(self.current_state.a, self.current_state.h + 1, RELEVANT)
            reward = (0, 0)
        self.current_state = new_state
        return new_state, reward    
        
    def getLegalActions(self):
        if (self.current_state.a == self.T) or (self.current_state.h == self.T):
            return [ADOPT]
        elif self.current_state.a > self.current_state.h:
            return [ADOPT, OVERRIDE, WAIT, MATCH]
        elif self.current_state.a == self.current_state.h:
            return [ADOPT, WAIT, MATCH]
        else:
            return [ADOPT, WAIT]
    
    def takeAction(self, action, rand_val):
        if action == 0:
            return self.getNextStateAdopt(rand_val)
        elif action == 1:
            return self.getNextStateOverride(rand_val)
        elif (action == 2) and (self.current_state.fork in [IRRELEVANT, RELEVANT]):
            return self.getNextStateWaitInactive(rand_val)
        else:
            return self.getNextStateWaitMatch(rand_val)

def main():
    env = Environment(alpha=0.35, T=9, gamma=0)
    print(env.current_state)

if __name__ == "__main__":
    main() 