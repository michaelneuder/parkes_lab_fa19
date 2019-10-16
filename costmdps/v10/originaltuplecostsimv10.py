from honestagentv10 import HonestAgent
from originaltuplecostenvv10 import Environment
from originaltuplecostmdpv10 import OriginalTupleMDP
import numpy as np
import progressbar as pb


alpha = 0.4
gamma = 0.5
T = 8

honest = 0
optimal = 1

if optimal:
    results = []
    for mining_cost in np.arange(0,1.1, 0.1):
    # for mining_cost in [0.5]:
        tuple_mdp = OriginalTupleMDP(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
        env = Environment(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)

        rho, opt_policy = tuple_mdp.solveWithPolicy()
        tuple_mdp.printPolicy(opt_policy)

        simulated_block_count = int(1e6)
        env.reset()
        bar = pb.ProgressBar()
        cumulative_reward = 0
        for _ in bar(range(simulated_block_count)):
            current_state = env.current_state
            action = tuple_mdp.getAction(opt_policy, current_state)
            new_state, reward = env.takeAction(action)
            cumulative_reward += tuple_mdp.evalRewardTuple(reward, rho)            
        print(mining_cost, cumulative_reward, cumulative_reward / simulated_block_count)
        results.append(cumulative_reward / simulated_block_count)
    print(results)

if honest:
    results = []
    for mining_cost in np.arange(0,1.1, 0.1):
    # for mining_cost in [0.5]:
        honest_agent = HonestAgent()
        env = Environment(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
        
        tuple_mdp = OriginalTupleMDP(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
        rho, opt_policy = tuple_mdp.solveWithPolicy()

        simulated_block_count = int(1e6)
        env.reset()
        bar = pb.ProgressBar()
        cumulative_reward = 0
        for _ in bar(range(simulated_block_count)):
            current_state = env.current_state
            action = honest_agent.act(current_state)
            new_state, reward = env.takeAction(action)
            cumulative_reward += honest_agent.evalRewardTuple(reward, rho)            
        print(mining_cost, cumulative_reward, cumulative_reward / simulated_block_count)
        results.append(cumulative_reward / simulated_block_count)
    print(results)