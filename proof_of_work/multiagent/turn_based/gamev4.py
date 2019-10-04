from environmentv4 import Environment
import matplotlib.pyplot as plt
import progressbar as pb


class Game(object):
    def __init__(self, mining_powers, T):
        self.number_players = len(mining_powers)
        self.env = Environment(mining_powers, T)

def main():
    game = Game([0.25, 0.15], 9)
    

    
if __name__ == "__main__":
    main()