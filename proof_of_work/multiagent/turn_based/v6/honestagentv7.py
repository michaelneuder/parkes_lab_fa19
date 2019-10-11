ADOPT = 0
OVERRIDE = 1
MINE = 2

class HonestAgent(object):
    def act(self, state): 
        a, h, fork = state
        if (a == 0) and (h == 0) and (fork == 0):
            return MINE
        if a > 0:
            return OVERRIDE
        elif h > 0:
            return ADOPT
        else:
            print(state)
            raise RuntimeError('invalid state for honest agent to reach')
    
    def step(self, state, action, reward, next_state):
        return

        