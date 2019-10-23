import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

opt_policy = np.asarray([
    [0, 0, 0, 0],
    [2, 2, 2, 0],
    [2, 1, 2, 0],
    [1, 1, 1, 0]
])
results = []
results_lens = []
bar = pb.ProgressBar()
for ii in bar(range(10000)):
    chain = ''
    alpha = 0.4
    a = 0
    h = 0
    for block in range(2016):
        rand_val = np.random.uniform()
        if rand_val < alpha:
            a += 1
        else:
            h += 1
        
        action = opt_policy[(a, h)]
        if action == 0:
            chain += 'h' * h
            a = 0
            h = 0
        elif action == 1:
            chain += 'a' * (h+1)
            a = a - h - 1
            h = 0
    results.append(chain.count('a')/len(chain))
    results_lens.append(len(chain))

plt.style.use('fivethirtyeight')
plt.hist(results, bins=100, color='g')
plt.axvline(0.41666412353515625, color='r', label='target', linewidth=0.4)
plt.axvline(np.mean(results), color='b', label='simulated', linewidth=0.4)
plt.legend(facecolor='white')
plt.xlabel(r'$\rho-$ relative rewards')
plt.show()
print(np.mean(results))

plt.cla()
plt.hist(results_lens, bins=100, color='purple')
plt.axvline(np.mean(results_lens), color='k', label='mean', linewidth=2.5)
plt.xlabel(r'$\ell-$ length of public chain ')
plt.legend(facecolor='white')
plt.show()
print(np.mean(results_lens))



