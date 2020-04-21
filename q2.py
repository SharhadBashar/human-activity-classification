import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import os
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sb
from sklearn.ensemble import AdaBoostClassifier

class Q2a():
    def __init__(self):
        self.CRange = [0.1, 1, 10, 100]
        df, df_A, df_B = self.getData()
        self.plot(df_A, df_B)
        self.createFolder()
        self.createData(df)
        results = self.run()
        C, bestAccuracy, intercept, coef, accuracies = self.getBestAccuracy(results, self.CRange)
        
        print('Best:', 'C:', C, 'Best Accuracy:', bestAccuracy, intercept, coef)
        print()
        print('Accuracy of C: 0.1:', np.mean(accuracies[0.1]) * 100, '%')
        print('Accuracy of C: 1:', np.mean(accuracies[1]) * 100, '%')
        print('Accuracy of C: 10:', np.mean(accuracies[10]) * 100, '%')
        print('Accuracy of C: 100:', np.mean(accuracies[100]) * 100, '%')

        self.plotBestDB(df, C, intercept, coef, bestAccuracy)
        
    def getData(self):
        df_A = pd.read_csv('classA.csv', names = ['att_1', 'att_2', 'class'])
        df_B = pd.read_csv('classB.csv', names = ['att_1', 'att_2', 'class'])
        df_A.loc[:, 'class'] = 1
        df_B.loc[:, 'class'] = 0
        df = pd.concat([df_A, df_B], ignore_index = True)

        df = df.sample(frac = 1)
        return df, df_A, df_B
    
    def plot(self, df_A, df_B):
        plt.scatter(df_A.iloc[:, 0], df_A.iloc[:, 1], c = 'red', label = 'A')
        plt.scatter(df_B.iloc[:, 0], df_B.iloc[:, 1], c = 'blue', label = 'B')
        plt.title('Q2 class data')
        plt.xlabel('Att 1')
        plt.ylabel('Att 2')
        plt.show()
    
    def createFolder(self, i = 0, n = 10):
        for i in range(n):
            if not os.path.exists('./data/q2/fold_' + str(i) + '/'):
                os.makedirs('./data/q2/fold_' + str(i) + '/')
        print('Done creatng folders')

    def createData(self, dataframe, i = 0, n = 10):
        for fold in range(n):
            for i in range(n):
                cv = KFold(n_splits = n, shuffle = True)
                for train_index, test_index in cv.split(dataframe):
                    train, test = dataframe.loc[train_index], dataframe.loc[test_index]
                    train.to_csv('./data/q2/fold_' + str(fold) + '/train_fold_' + str(i) + '.csv', index = False, header = False)
                    test.to_csv('./data/q2/fold_' + str(fold) + '/test_fold_' + str(i) + '.csv', index = False, header = False)
        print('Done creating data for Q2')
    
    @ignore_warnings(category=ConvergenceWarning)
    def svm(self, df_train, df_test, C):
        clf = LinearSVC(random_state = 0, tol = 1e-6, C = C, max_iter = 1e6)
        clf.fit(df_train.iloc[:, :2], df_train.iloc[:, 2])
        train_acc = clf.score(df_train.iloc[:, :2], df_train.iloc[:, 2])
        test_acc = clf.score(df_test.iloc[:, :2], df_test.iloc[:, 2])
        return [clf.coef_[0], clf.intercept_, train_acc, test_acc]
    
    def getBestAccuracy(self, results, CRange, n = 10):
        C = -1
        bestAccuracy = 0
        intercept = 0
        coef = [0, 0]
        accuracies = {0.1: [], 1: [], 10: [], 100: []}
        for i in range(len(CRange) * n * n):
            accuracy = results[i][0]['test_acc']
            accuracies[results[i][0]['C']].append(accuracy)
            if accuracy > bestAccuracy:
                C = results[i][0]['C']
                bestAccuracy = accuracy
                intercept = results[i][0]['intercept']
                coef = results[i][0]['coef']
        return [C, bestAccuracy, intercept, coef, accuracies]
    
    
    def plotBestDB(self, df, C, intercept, coef, bestAccuracy):
        colors = ['red', 'blue']
        att_1 = df.iloc[:, 0]
        att_2 = df.iloc[:, 1]
        label = df.iloc[:, 2]
        att_1_range = np.linspace(np.min(att_1), np.max(att_1), 100)
        att_2_val = (-intercept - att_1_range * coef[0]) / coef[1]
        plt.scatter(att_1, att_2, c = label, cmap = matplotlib.colors.ListedColormap(colors))
        plt.plot(att_1_range, att_2_val, '-r', label = ('Accuracy:', bestAccuracy * 100))
        plt.title('C: ' + str(C))
        plt.xlabel('Att 1')
        plt.ylabel('Att 2')

        plt.show()
        
    def run(self):
        results = []
        CRange = self.CRange
        start = time.time()
        for C in CRange:
            for fold in range(10):
                for i in range(10):
                    df_train = pd.read_csv('./data/q2/fold_' + str(fold) + '/train_fold_' + str(i) + '.csv')
                    df_test = pd.read_csv('./data/q2/fold_' + str(fold) + '/test_fold_' + str(i) + '.csv')
                    coef, intercept, train_acc, test_acc = self.svm(df_train, df_test, C)
                    result = {
                        "C": C,
                        "coef": coef,
                        "intercept": intercept,
                        "train_acc": train_acc, 
                        "test_acc": test_acc
                    }
                    results.append([result])
                print('Done Fold:', fold + 1, 'out of 10')
            print('Done C:', C)
        end = time.time()
        print((end - start) / 60)
        return results


class Q2b():
    def __init__(self):
        self.n = 10
        self.T = 50
        self.sample = 100
        self.C = 1
        self.clfs = []
        self.betas = []
        
        self.getData()
        self.adaboost()
        self.betas = np.array(self.betas)
        self.predict()
        self.plot()
        print('Accuracy:', self.calcScore(), '%')
        
    
    def getData(self):
        df_A = pd.read_csv('classA.csv', names = ['att_1', 'att_2', 'class'])
        df_B = pd.read_csv('classB.csv', names = ['att_1', 'att_2', 'class'])
        df_A.loc[:, 'class'] = 1
        df_B.loc[:, 'class'] = 0
        df = pd.concat([df_A, df_B], ignore_index = True)
        df = df.sample(frac = 1)
        self.trainX = df.iloc[:, :-1]
        self.trainY = df.iloc[:, -1]
    
    def predict(self, preds = []):
        T = self.T
        betas = self.betas
        clfs = self.clfs
        alphas = 0.5 * np.log(1 / betas)
        for i in range(T):
            pred = clfs[i].predict(self.trainX)
            preds.append(alphas[i] * pred)
        preds = np.array(preds)
        self.finalPred = np.array(np.sign(np.sum(preds, axis = 0)))
    
    def calcScore(self):
        socre = 0
        n = len(self.trainY)
        for i in range(n):
            if (self.trainY.iloc[i] == self.finalPred[i]):
                socre += 1
        return self.score * 100
    
    def plot(self):
        trainX = self.trainX
        trainY = self.trainY
        clf = AdaBoostClassifier(n_estimators = 100, random_state = 0)
        clf.fit(trainX, trainY)
        self.score = clf.score(trainX, trainY)
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        x_min, x_max = trainX.iloc[:, 0].min() - .5, trainX.iloc[:, 0].max() + .5
        y_min, y_max = trainX.iloc[:, 1].min() - .5, trainX.iloc[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
        
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax = plt.gca()
        ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5)
        ax.contour(xx, yy, Z,  2, cmap='RdBu')
        ax.scatter(trainX.iloc[:,0], trainX.iloc[:,1], c = trainY, cmap = cm_bright)
        
        plt.show()
        
    def adaboost(self):
        C = self.C
        T = self.T
        sample = self.sample
        trainX = self.trainX
        trainY = self.trainY
        m = len(trainX)
        D = np.ones(m) / m
        while (len(self.clfs) < T):
            i = np.random.choice(m, size = sample)
            x = trainX.iloc[i]
            y = trainY.iloc[i]
            d = D[i]
            clf = SVC(kernel = 'linear', C = C, tol = 1e-8)
            clf.fit(x, y, sample_weight = d)
            pred = clf.predict(trainX)
            error = np.sum(D[np.where(pred != trainY)])
            if error < 0.5:
                beta = error / (1 - error)
                for j in range(len(D)):
                    if (pred[j] == trainY[j]): D[j] = beta
                    else: D[j] = 1
                D = D / np.sum(D)
                self.betas.append(beta)
                self.clfs.append(clf)
        
Q2a()
Q2b()