class HonestAgent(object):
    def act(self, state): 
        if state[0] > 0:
            return 'override'
        elif state[1] > 0:
            return 'adopt'
        else:
            print(state)
            raise RuntimeError('invalid state for honest agent to reach')

        