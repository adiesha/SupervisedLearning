import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sympy.strategies.core import switch


def crossValidation(D,K,clssifierMethod):
    N = len(D.index)  # Number of entries
    print(str(K)+'-fold cross validation on '+str(N)+' data points')

    D = pd.DataFrame(shuffle(D.values))     # Shuffling the dataset

    Theta = np.zeros(K)

    for i in range(K):

        print('From '+str(i*np.floor(N/K)) + ', to ' + str((i+1)*np.floor(N/K)-1) + ' as test set')
        D_test = D.iloc[int(i*np.floor(N/K)):int((i+1)*np.floor(N/K)-1), :].copy().reset_index(drop=True)
        D_train = D[~D.index.isin(range(int(i*np.floor(N/K)), int((i+1)*np.floor(N/K)-1)))].copy().reset_index(drop=True)

        if clssifierMethod == 'Dtree':
            result =

        print()

def main():
    print("Hello World")

    data = pd.read_csv('iris.data', header=None)
    N = len(data.index)  # Number of entries

    K = 5       # number of folds in K-fold
    clssifierMethod = 'Dtree'

    crossValidation(data.copy(), K, clssifierMethod)

if __name__ == "__main__":
    main()