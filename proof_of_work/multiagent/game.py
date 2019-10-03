from environmentv3 import Environment
from honest_agent import HonestAgent
import progressbar as pb


class Game(object):
    def __init__(self, mining_powers, T):
        self.number_players = len(mining_powers)
        self.env = Environment(mining_powers, T)

        # two honest agents
        self.honest_agent_1 = HonestAgent()
        self.honest_agent_2 = HonestAgent()

    def stepTime(self):
        self.env.honestMine()
        action = self.honest_agent_1.act(self.env.getState(0))
        self.takeActionPlayer(0, action)
        action = self.honest_agent_2.act(self.env.getState(1))
        self.takeActionPlayer(1, action)

    def takeActionPlayer(self, player_index, action):
        if action == 'adopt':
            self.env.playerAdopt(player_index)
        elif action == 'override':
            self.env.playerOverride(player_index)
        # mining
        else:
            self.env.playerMine(player_index)
    
    def play(self):
        bar = pb.ProgressBar()
        for _ in bar(range(10000)):
            self.stepTime()


def main():
    game = Game([0.25, 0.15], 9)
    game.play()
    print(game.env.chain)

if __name__ == "__main__":
    main()