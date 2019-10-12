from matchmdpv8 import MatchMDP
import numpy as np

alpha = 0.4
gamma = 0.5
T = 8

for mining_cost in np.arange(0.7, 1.01, 0.01):
    match_mdp = MatchMDP(alpha=alpha, gamma=gamma, T=T, mining_cost=mining_cost)
    policy = match_mdp.solveWithPolicy()
    match_mdp.printPolicy(policy)
    opt_value = match_mdp.solveWithValue()
    print('{:.05f}'.format(float(opt_value)))
    

