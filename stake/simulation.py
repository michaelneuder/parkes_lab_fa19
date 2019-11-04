import numpy as np
import progressbar as pb


def getCurrentState(alpha):
    k_0 = np.random.choice([1,2,3], p=[alpha, alpha, 1 - 2*alpha])
    k_1 = np.random.choice([1,2,3], p=[alpha, alpha, 1 - 2*alpha])
    e = np.random.binomial(32, p=alpha)
    return (k_0, k_1, e)

for alpha in np.arange(0, 0.5, 0.025):
    chain = ''
    bar = pb.ProgressBar()
    for minute in bar(range(525600)):
        k_0, k_1, e = getCurrentState(alpha)
        if (k_0 == 2) and (k_1 == 1) and (e > 16):
            chain += 'ca'
        else:
            if k_0 == 1:
                chain += 'a'
            else:
                chain += 'h'

    print('')
    print('alpha:', alpha)
    print('a: ', chain.count('a'))
    print('c: ', chain.count('c'))
    print('h: ', chain.count('h'))

