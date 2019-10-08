ADOPT = 0
OVERRIDE = 1
WAIT = 2

class HonestAgent(object):
    def act(self, state, eps): 
        if state[0] > 0:
            return OVERRIDE
        elif state[1] > 0:
            return ADOPT
        else:
            print(state)
            raise RuntimeError('invalid state for honest agent to reach')
    
    def step(self, state, action, reward, next_state):
        return

        