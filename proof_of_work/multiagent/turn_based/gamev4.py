from environmentv4 import Environment
from honestagentv4 import HonestAgent
from semiselfishagentv4 import SemiSelfishAgent
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb


class Game(object):
    def __init__(self, mining_powers, selfish_miners, T):
        self.env = Environment(mining_powers, T)
        self.mining_powers = mining_powers

        # agents
        assert(len(mining_powers) == len(selfish_miners))
        self.agents = []
        for miner in selfish_miners:
            if miner:
                self.agents.append(SemiSelfishAgent())
            else:
                self.agents.append(HonestAgent())
        print(self.agents)
        # results
        self.results = []

    def play(self, block_count=int(1e4)):
        bar = pb.ProgressBar()
        for i in bar(range(block_count)):
            self.runTurn()
            if i % 100 == 1:
                chain = self.env.chain
                percentages = []
                for j in range(len(self.mining_powers)):
                    percentages.append(chain.count(str(j)) / i)
                self.results.append(percentages)
        
    def runTurn(self):
        winner = self.env.getNextBlockWinner()
        winner_state = self.env.getState(winner)
        action = self.agents[winner].act(winner_state)
        self.takeActionPlayer(winner, action)
        if action == 'override':
            losers = list(range(len(self.agents)))
            losers.remove(winner)
            for loser in losers:
                loser_state = self.env.getState(loser)
                action = self.agents[loser].act(loser_state)
                self.takeActionPlayer(loser, action)
                
    def takeActionPlayer(self, player_index, action):
        if action == 'adopt':
            self.env.adopt(player_index)
        elif action == 'override':
            self.env.override(player_index)
    
    def plotRewards(self):
        rewards = np.asarray(self.results)
        _f, ax = plt.subplots(figsize=(10,5))
        for i in range(len(self.mining_powers)):
            ax.plot(rewards[:,i])
            ax.axhline(self.mining_powers[i], color='k', alpha=0.25)
        ax.set_xlabel('number of blocks mined', size=20)
        ax.set_ylabel('percentage of reward', size=20)
        plt.show()

def main():
    game = Game([0.65, 0.35], [0, 1], 9)
    game.play(block_count=int(1e5))
    chain = game.env.chain
    print(chain)
    print('honest0', chain.count('0')/len(chain))
    print('honest1', chain.count('1')/len(chain))
    print('honest2', chain.count('2')/len(chain))
    
    game.plotRewards()


if __name__ == "__main__":
    main()