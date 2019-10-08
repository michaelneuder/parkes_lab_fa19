from collections import deque
from environmentv4 import Environment
from honestagentv4 import HonestAgent
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
np.random.seed(0)
import progressbar as pb
from selfishagentv4 import SelfishAgent


class Game(object):
    def __init__(self, mining_powers, selfish_miners, gammas, T, block_count):
        self.env = Environment(mining_powers, gammas, T)
        self.mining_powers = mining_powers
        self.selfish_miners = selfish_miners
        self.block_count = block_count

        # agents
        assert(len(mining_powers) == len(selfish_miners))
        self.agents = []
        for miner in selfish_miners:
            if miner:
                self.agents.append(SelfishAgent(T))
            else:
                self.agents.append(HonestAgent())
        # results
        self.results = []

    def play(self):
        bar = pb.ProgressBar()
        for i in bar(range(self.block_count)):
            self.runTurn(0)
            # if i % 100 == 1:
            #     self.collectResults(i)
    
    def collectResults(self, iteration_number):
        chain = self.env.chain
        percentages = []
        for j in range(len(self.mining_powers)):
            percentages.append(chain.count(str(j)) / len(chain))
        self.results.append(percentages)
        
    def runTurn(self, verbose=0):
        winner = self.env.getNextBlockWinner()
        winner_state = self.env.getState(winner)
        winner_action = self.agents[winner].act(winner_state)
        if verbose: 
            print(self.env.chain)
            print(winner, winner_state, winner_action)
        self.takeActionPlayer(winner, winner_action)
        if winner_action == 'override':
            turn_queue = deque()
            [turn_queue.append(i) for i in range(len(self.mining_powers)) if i != winner]
            while len(turn_queue):    
                current_player = turn_queue.pop()
                current_player_state = self.env.getState(current_player)
                current_player_action = self.agents[current_player].act(current_player_state)
                if verbose: print(current_player, current_player_state, current_player_action)
                self.takeActionPlayer(current_player, current_player_action)
                if current_player_action == 'override':
                    [turn_queue.append(i) for i in range(len(self.mining_powers)) if i != current_player]
            if verbose: print(self.env.chain)
                
    def takeActionPlayer(self, player_index, action):
        if action == 'adopt':
            self.env.adopt(player_index)
        elif action == 'override':
            self.env.override(player_index)
        elif action == 'match':
            self.env.match(player_index)
    
    def plotRewards(self):
        rewards = np.asarray(self.results)
        _f, ax = plt.subplots(figsize=(10,5))
        ax.set_ylim([0,1])
        for i in range(len(self.mining_powers)):
            if self.selfish_miners[i]:
                label = r'selfish miner: $\alpha={}$'.format(self.mining_powers[i])
            else:
                label = r'honest miner: $\alpha={}$'.format(self.mining_powers[i])
            ax.plot(list(range(1, self.block_count, 100)), rewards[:,i], label=label)
            ax.axhline(self.mining_powers[i], color='k', alpha=0.25)
        ax.set_xlabel('number of blocks mined', size=20)
        ax.set_ylabel('percentage of reward', size=20)
        plt.legend()
        plt.show()

def main():
    selfish_rewards = []
    powers = np.arange(0.025, 0.5, 0.025)
    for selfish_power in [0.45]:
        game = Game([1 - selfish_power, selfish_power], [0, 1], [1, 0], T=9, block_count=int(3e5))
        game.play()
        chain = game.env.chain
        for player in range(len(game.mining_powers)):
            reward = chain.count(str(player)) / len(chain)
            proportional_rewards = reward / game.mining_powers[player]
            if game.selfish_miners[player]:
                print('selfish miner: alpha={} -- reward: {:.04f} -- earned: {:.04f}'.format(
                    game.mining_powers[player], reward, proportional_rewards))
                selfish_rewards.append(reward)
            else:
                print('honest miner: alpha={} -- reward: {:.04f} -- earned: {:.04f}'.format(
                    game.mining_powers[player], reward, proportional_rewards))
    print(selfish_rewards)
    _f, ax = plt.subplots(figsize=(10,7))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 0.5, 0.05))
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_xlabel(r'$\alpha -$ fraction of hashrate', size=20)
    ax.set_ylabel('revenue', size=20)
    ax.plot(powers, selfish_rewards, 'bo-', label='selfish mining')
    ax.plot(powers, powers, 'r--', label='honest mining')
    ax.plot(powers, powers / (1-powers), 'g', linestyle='dotted', label=r'$\alpha \;/ \;(1 - \alpha)$')
    plt.legend()
    plt.show()

    # game.plotRewards()


if __name__ == "__main__":
    main()