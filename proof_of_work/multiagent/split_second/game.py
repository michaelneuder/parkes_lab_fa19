from environmentv3 import Environment
from honest_agent import HonestAgent
import matplotlib.pyplot as plt
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
        self.results_honest = []
        self.results_selfish1 = []
        self.results_selfish2 = []
        bar = pb.ProgressBar()
        for i in bar(range(int(1e6))):
            if i % 1000 == 0:
                chain = self.env.chain
                self.results_honest.append(chain.count('h')/len(chain))
                self.results_selfish1.append(chain.count('0')/len(chain))
                self.results_selfish2.append(chain.count('1')/len(chain))
            self.stepTime()
    
    def plotRewards(self):
        _f, ax = plt.subplots(figsize=(10,5))
        ax.plot(self.results_honest, label=r'honest -- $\alpha$=0.6')
        ax.plot(self.results_selfish1, label=r'honest1 -- $\alpha$=0.25')
        ax.plot(self.results_selfish2, label=r'honest2 -- $\alpha$=0.15')
        ax.set_xlabel('number of blocks mined', size=20)
        ax.set_ylabel('percentage of reward', size=20)
        ax.axhline(0.6, color='k', alpha=0.25)
        ax.axhline(0.25, color='k', alpha=0.25)
        ax.axhline(0.15, color='k', alpha=0.25)
        plt.legend()
        plt.show()


def main():
    game = Game([0.25, 0.15], 9)
    game.play()
    chain = game.env.chain
    print(chain)
    print('honest', chain.count('h'), chain.count('h')/len(chain))
    print('selfish1', chain.count('0'), chain.count('0')/len(chain))
    print('selfish2', chain.count('1'), chain.count('1')/len(chain))
    game.plotRewards()

if __name__ == "__main__":
    main()