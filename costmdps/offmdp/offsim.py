from costmdp import CostMDP
from offenv import Environment
from offmdp import OffMDP
import numpy as np
import progressbar as pb


alpha = 0.4
gamma = 0.5
T = 8

results = []
for mining_cost in np.arange(0,1.1, 0.1):
    match_mdp = CostMDP(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
    env = Environment(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)

    opt_policy = match_mdp.solveWithPolicy()
    match_mdp.printPolicy(opt_policy)

    simulated_block_count = int(1e6)
    env.reset()
    bar = pb.ProgressBar()
    cumulative_reward = 0
    extra_actions = 0
    for _ in bar(range(simulated_block_count)):
        current_state = env.current_state
        action = match_mdp.getAction(opt_policy, current_state)
        MINE = 2
        MATCH = 3
        if action == MATCH:
            new_state, reward = env.takeAction(action)
            cumulative_reward += reward
            action = MINE
            new_state, reward = env.takeAction(action)
            cumulative_reward += reward
            extra_actions += 1
        else:
            new_state, reward = env.takeAction(action)
            cumulative_reward += reward
            

    print(cumulative_reward, matches,  cumulative_reward / (simulated_block_count + extra_actions) )
    print('-----')
    print(mining_cost, cumulative_reward, cumulative_reward / simulated_block_count)
    results.append(cumulative_reward / simulated_block_count)

print(results)
