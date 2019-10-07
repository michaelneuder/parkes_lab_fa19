import numpy as np

class SelfishAgent(object):
    def __init__(self):
        self.policy = np.asarray([
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [2, 2, 2, 0, 0, 0, 0, 0, 0], 
            [2, 1, 2, 2, 2, 0, 0, 0, 0], 
            [2, 2, 1, 2, 2, 2, 0, 0, 0], 
            [2, 2, 2, 1, 2, 2, 2, 0, 0], 
            [2, 2, 2, 2, 1, 2, 2, 2, 0], 
            [2, 2, 2, 2, 2, 1, 2, 2, 0], 
            [2, 2, 2, 2, 2, 2, 1, 2, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

    def act(self, state): 
        action = self.policy[state]
        if action == 0:
            return 'adopt'
        if action == 1:
            return 'override'
        return 'wait'

        