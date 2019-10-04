class SemiSelfishAgent(object):
    def act(self, state): 
        a, h = state
        if h > a:
            return 'adopt'
        if (h == a-1) and (h > 0):
            return 'override'
        return 'wait'

        