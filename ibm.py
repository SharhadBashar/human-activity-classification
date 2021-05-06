import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix

#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.neural_network as nn
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.cluster import KMeans


class IBM():
    def __init__(self):
        self.n = 10
        self.classifiers = ['AdaBoost', 'Gradient Boosting', 'Random Forrest', 'SVM', 
                            'Decision Tree', 'Bagging', 'kNN', 'GausianNB', 'MLPC', 'KMeans']
        self.train, self.test = self.getData()
        self.eda()
        self.elbow(self.train)
        modelAccuracies = self.comparison()
        self.plotBarAccuracies(modelAccuracies)
        
    def getData(self):
        train = shuffle(pd.read_csv('train.csv'))
        train = train.replace(to_replace = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'], 
                           value = [0, 1, 2, 3, 4, 5])
        train = train.drop(['subject'], axis = 1)

        test = pd.read_csv('test.csv')
        test = test.replace(to_replace = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'], 
                           value = [0, 1, 2, 3, 4, 5])
        test = test.drop(['subject'], axis = 1)
        return [train, test]
    
    def seperateData(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
    
    def eda(self, eda = pd.read_csv('train.csv'), edaTest = pd.read_csv('test.csv')):
        fig = sb.countplot(x = 'Activity' , data = eda)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Entries per class Training Data')
        plt.show()
        
        fig = sb.countplot(x = 'Activity' , data = edaTest)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Entries per class Testing Data')
        plt.show()

        sb.boxplot(x = 'Activity', y = 'tBodyAccMag-mean()', data = eda, showfliers=False)
        plt.ylabel('Body Acceleration Magnitude mean')
        plt.title('Boxplot of Body Acceleration Magnitude mean for each class')
        plt.show()

        edaX = eda.drop(['subject', 'Activity'], axis = 1)
        tsne = TSNE(verbose = 1, perplexity = 50).fit_transform(edaX)
        sb.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue = eda['Activity'])
        plt.title('t-SNE of each class')
        plt.show()

        tsne = TSNE(verbose = 1, perplexity = 50).fit_transform(edaX)
        sb.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue = eda['subject'])
        plt.title('t-SNE of each subject')
        plt.show()
    
    def elbow(self, data):
        inertia_list = list()
        k_list = list(range(1, 11))
        for k in k_list:
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(data)
            inertia_list.append(kmeans.inertia_)
        plt.plot(k_list, inertia_list)
        plt.scatter(k_list, inertia_list)
        plt.xlabel('K values')
        plt.ylabel('Inertia')
        plt.show()

    def plotBarAccuracies(self, accuracies):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(self.classifiers, accuracies)
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracies')
        plt.title('Accuracy for each classifier')
        plt.show()
    
    def accuracy(self, true, pred):
        return accuracy_score(true, pred) * 100
    
    def plotBarAccuracies(self, accuracies):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(self.classifiers, accuracies)
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracies')
        plt.title('Accuracy for each classifier')
        plt.show()
        
    def comparison(self):
        accuracies = []
        trainX, trainY = self.seperateData(self.train)
        testX, testY = self.seperateData(self.test)
        start = time.time()
        
        #1.AdaBoost
        ab = AdaBoostClassifier(LogisticRegression(), n_estimators = 200)
        ab.fit(trainX, trainY)
        predab = ab.predict(testX)
        accuracies.append(self.accuracy(testY, predab))
        print('Adaboost')
        
        #2.Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators = 200)
        gb.fit(trainX, trainY)
        predgb = gb.predict(testX)
        accuracies.append(self.accuracy(testY, predgb))
        print('Gradient Boosting')
            
        #3.Random Forest
        rf = RandomForestClassifier(n_estimators = 200)
        rf.fit(trainX, trainY)
        predrf = rf.predict(testX)
        accuracies.append(self.accuracy(testY, predrf))
        print('Random Forrest')
        
        #4.SVM
        ovr = OneVsRestClassifier(svm.SVC(kernel = 'linear', probability = True, random_state = 0))
        ovr.fit(trainX, trainY)
        predovr = ovr.predict(testX)
        accuracies.append(self.accuracy(testY, predovr))
        print('SVM')
        
        #5.Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(trainX, trainY)
        preddt = dt.predict(testX)
        accuracies.append(self.accuracy(testY, preddt))
        print('Decision Trees')
        
        #6.Bagging
        bag = BaggingClassifier(n_estimators = 200)
        bag.fit(trainX, trainY)
        predbag = bag.predict(testX)
        accuracies.append(self.accuracy(testY, predbag))
        print('Bagging')
        
        #7.kNN
        knn = KNeighborsClassifier()
        knn.fit(trainX, trainY)
        predknn = knn.predict(testX)
        accuracies.append(self.accuracy(testY, predknn))
        print('kNN')
        
        #8.gausianNB
        gnb = GaussianNB()
        gnb.fit(trainX, trainY)
        predgnb = gnb.predict(testX)
        accuracies.append(self.accuracy(testY, predgnb))
        print('Gaussian Naive Bayes')
        
        #9.MLPC
        mlpc = nn.MLPClassifier(hidden_layer_sizes = (50, 50), max_iter = 1000, early_stopping = True, random_state = 0)
        mlpc.fit(trainX, trainY)
        predmlpc = dt.predict(testX)
        accuracies.append(self.accuracy(testY, predmlpc))
        print('Multi-layer Perceptron classifier')
        
        #10. KMeans
        kmeans = KMeans(n_clusters = 6)
        kmeans.fit(trainX)
        predKMeans = kmeans.labels_
        accuracies.append(self.accuracies(trainY, predKMeans))
        print('KMeans')

        print('Time taken:', (time.time() - start) / 60, 'min')
        print(accuracies)
        
        return accuracies
IBM()
