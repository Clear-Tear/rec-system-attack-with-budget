from scipy.stats import boxcox

import numpy as np

x = np.loadtxt("rec_attack_budget/tti.txt")
x = -x
for i in range(100):
    if x[i] < 0 :
        x[i] = 0.000000000001  

y,lambda0 = boxcox(x,lmbda = None, alpha = None)

print(lambda0)



