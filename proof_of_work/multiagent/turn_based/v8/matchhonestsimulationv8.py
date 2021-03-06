from environmentv8 import Environment
from honestagentv8 import HonestAgent
import numpy as np
import progressbar as pb


alpha = 0.4
gamma = 0.5
T = 8
honest_agent = HonestAgent()

results = []
for mining_cost in np.arange(0,1.1, 0.1):
    env = Environment(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
    
    simulated_block_count = int(1e6)
    env.reset()
    bar = pb.ProgressBar()
    cumulative_reward = 0
    for _ in bar(range(simulated_block_count)):
        current_state = env.current_state
        action = honest_agent.act(current_state)
        new_state, reward = env.takeAction(action)
        cumulative_reward += reward

    print(mining_cost, cumulative_reward, cumulative_reward / simulated_block_count)
    results.append(cumulative_reward / simulated_block_count)

print(results)
