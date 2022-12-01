import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from random import seed
from random import randint


def stochasticGradientDescent(x, y, theta, alpha, m, numIters):
    calculatedCosts = []
    i = random.randint(0, len(dependent) - 1)
    for k in range(0, numIters):
        prediction= 0
        for j in range(10):
            prediction = prediction + (x[i, j]* thetas[j])
            newtheta = thetas[j] - alpha * (x[i, j]* (x[i, j]* thetas[j] - y[i]))
            theta[j] = newtheta
        cost = ((prediction - y[i]) ** 2) / (2 * m)
        calculatedCosts.append(cost)
        print(f"===== Iteration ({k}) ======")
        print(theta)
        print(cost)
    return theta, calculatedCosts


data = pd.read_csv("communities.data", header=None, sep=",", engine='python')
print(data)
df1 = data.loc[:, [127]]
print(type(df1))
dependent = df1.to_numpy()
print(type(dependent))
random_obs= randint(0, 1994)
df2 = data.loc[:, [17, 26, 27, 31, 32, 37, 76, 90, 95]]
print(type(df2))
independentVars = df2.to_numpy()
print(type(independentVars))

onesColumn = np.ones((1994, 1))
independentVars = np.hstack((onesColumn, independentVars))
alpha = 0.1
iniThetas = []
for i in range(0, independentVars.shape[1]):
    iniThetas.append( np.random.rand() )
thetas = np.array(iniThetas)
numIters = 500

estimatedThetas, costs = stochasticGradientDescent(independentVars, dependent, thetas, alpha, 1994, numIters)
print(">>> Estimated thetas")
print(estimatedThetas)

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(list(range(numIters)), costs, '-r')
plt.show()

