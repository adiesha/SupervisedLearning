import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sympy.strategies.core import switch
import decisiontree as dt
import Performance as perf
import knn

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

            listofclusters = D_train[4].unique()
            gTruthCol = 4
            predCol = 5
            listofattributes = [0, 1, 2, 3]
            neta = 5
            phi = 0.9

            node = dt.createdecisionTree(D_train, neta, phi, listofattributes, gTruthCol, listofclusters)
            result = node.predict_data_set(D_test)
            # print(result)

            Theta[i] = perf.F_measure(result, gTruthCol, predCol)

        elif clssifierMethod == 'KNN':
            predictions = []
            gTruthCol = 4
            predCol = 5
            listofattributes = [0, 1, 2, 3]

            k = 3
            for j in range(len(D_test)):
                tmp = knn.k_nearest_neighbors(D_train.iloc[:, 0:4].to_numpy(), D_train.iloc[:, gTruthCol].to_numpy(), D_test.iloc[j, 0:4].to_numpy(), k)
                predictions.append(tmp)

            # acc = (label_test == predictions).sum() / len(predictions)
            # print('Accuracy : ', + acc)

            result = pd.DataFrame(D_test)
            # df.insert(4, '4', label_test)
            result.insert(predCol, 'Results', predictions)

            Theta[i] = perf.F_measure(result, gTruthCol, predCol)
    print('Performance measure in each fold')
    print(str(Theta))

    mu_Theta = np.mean(Theta)
    var_Theta = np.var(Theta)
    print('Mean performance measure' + str(mu_Theta))
    print('Variance performance measure' + str(mu_Theta))

    return mu_Theta, var_Theta


def main():
    print("Hello World")

    data = pd.read_csv('iris.data', header=None)
    N = len(data.index)  # Number of entries

    K = 5       # number of folds in K-fold
    clssifierMethod = 'KNN'


    crossValidation(data.copy(), K, clssifierMethod)

if __name__ == "__main__":
    main()