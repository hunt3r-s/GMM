import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix



def fit(clusterCt, iterations, X):

    rowDim, colDim = X.shape
    phi = np.full(shape=clusterCt, fill_value=1/clusterCt)
    weight = np.full(shape=X.shape, fill_value=1/clusterCt)
    randRow = np.random.randint(low=0, high=rowDim, size=clusterCt)
    mu = [X[row, : ] for row in randRow]
    sigma = [np.cov(X.T) for _ in range(clusterCt)]

    for iter in range(iterations):
        # call EM steps
        weight, phi = expectation(X, weight, clusterCt, rowDim, mu, sigma, phi)
        mu, sigma = Maximization(X, clusterCt, weight, mu, sigma)

    return mu, sigma

# Expectation step
def expectation(X, weight, clusterCt, rowDim, mu, sigma, phi):

    weight = logProb(X, clusterCt, rowDim, mu, sigma, phi)
    phi = weight.mean(axis=0)
    return weight, phi

def Maximization(X, clusterCt, weight, mu, sigma):

    for i in range(clusterCt):
        iweight = weight[:, [i]]
        total = iweight.sum()
        mu[i] = (X * iweight).sum(axis=0) / total
        sigma[i] = np.cov(X.T, aweights=(iweight/total).flatten(), bias=True)

    return mu, sigma

def logProb(X, clusterCt, rowDim, mu, sigma, phi):
    prob = np.zeros((rowDim, clusterCt))
    for i in range(clusterCt):
        mvd = multivariate_normal(mean=mu[i], cov=sigma[i])
        prob[:, i] = mvd.pdf(X)

    weight = (prob * phi) / (prob * phi).sum(axis=1)[:, np.newaxis]
    return weight

def plotGMM(X, pairs):
    rows = len(pairs) // 2
    cols = 2

    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(pairs):
        plt.subplot(rows, cols, i+1)
        plt.title("GMM")
        plt.xlabel("A")
        plt.ylabel("B")
        plt.scatter(X[:, x] + np.random.uniform(low=-0.05, high=0.05, size=X[:, x].shape),
                    X[:, y] + np.random.uniform(low=-0.05, high=0.05, size=X[:, x].shape))
        plt.tight_layout()
        plt.show()


data = pd.read_csv('freshman_kgs.csv')
data = data.fillna(data.mean())
data =data._get_numeric_data()
X = data.to_numpy()
print(fit(clusterCt = 4, iterations = 50, X = X))

#plotGMM(X=X, pairs=[
#        (0,1), (2,3),
#        (0,2), (1,3) ])

#for i in range(np.size(X, 1)-1):
    #print(covs[i])
    #means[i], covs[i] = fit(clusterCt = 4, iterations = 50, tolerance = 1e-3, seed = 4, X = X[:, [i, i+1]])

#fit(clusterCt = 4, iterations = 50, X = X[:, [0, 1]])
# means, covs = fit(clusterCt = 4, iterations = 50, tolerance = 1e-3, seed = 4, X = X[:, [0, 1]])
#print(means.type())
#print(covs.type())

# plotGMM(X[:, [0, 1]], means, covs)