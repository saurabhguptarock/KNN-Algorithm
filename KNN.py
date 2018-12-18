import numpy as np
import pandas as pd

dfx = pd.read_csv('xdata.csv')
dfy = pd.read_csv('ydata.csv')

X = dfx.values
Y = dfy.values

X = X[:, 1:]
Y = Y[:, 1:].reshape((-1,))

query_x = np.array([2, 3])


def distance(x1, x2):
    return np.sqrt(sum(x1 - x2)**2)


def knn(X, Y, query_point, k=10):
    vals = []
    m = X.shape[0]

    for i in range(m):
        d = distance(query_point, X[i])
        vals.append((d, Y[i]))

    vals = sorted(vals)
    vals = np.array(vals[:k])
    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred


pred = int(knn(X, Y, query_x))
print(pred)
