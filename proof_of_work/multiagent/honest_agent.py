class HonestAgent(object):
    def act(self, state):
        # we found a block so publish it
        if state[2] != 0:
            return 'override'
        # someone else found a block so adopt it
        elif state[0] != state[1]:
            return 'adopt'
        else:
            return 'mine'
