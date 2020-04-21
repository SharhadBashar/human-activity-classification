import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.svm import LinearSVC

class Q1():
	def __init__(self):
		self.randomState = 0
		self.tolerance = 1e-10
		self.maxIters = 1e6
		self.C = [0.001, 0.01, 0.1, 1, 10]
		#name_attributenumber_datasetnumber
		self.dataset_1 = {
			'name': 'HW 3 Dataset 1',
			'filename': 'hw3_dataset1.csv',
			'col_names': ['att_1_1', 'att_2_1', 'label_1']
		}
		self.dataset_2 = {
			'name': 'HW 3 Dataset 2',
		    'filename': 'hw3_dataset2.csv',
		    'col_names': ['att_1_2', 'att_2_2', 'label_2']
		}
		self.color = ['red', 'blue']
		self.run()

	def __str__(self):
		return 'Q1: Linear SVM for Two-class Problem'

	def readData(self, data):
		df = pd.read_csv(data['filename'], names = data['col_names'])
		return df

	#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
	def svm(self, df, C):
		X = df.iloc[:, :2]
		y = df.iloc[:, 2]
		clf = LinearSVC(random_state = self.randomState, tol = self.tolerance, C = C, max_iter = self.maxIters)
		clf.fit(X, y)
		return clf.score(X, y), clf.coef_[0], clf.intercept_

	def plotScatter(self, data, df):
		att_1 = df.iloc[:, 0]
		att_2 = df.iloc[:, 1]
		label = df.iloc[:, 2]
		plt.scatter(att_1, att_2, c = label, cmap = matplotlib.colors.ListedColormap(self.color))
		plt.title(data['name'])
		plt.xlabel('Att 1')
		plt.ylabel('Att 2')
		plt.show()

	def plotLine(self, i, data, df, CResults):
		# att1 * coef [0] + att2 * coef [1] + interc = 0 
		# att2 = (-interc - att1 * coef [0]) / coef [1]
		att_1 = df.iloc[:, 0]
		att_2 = df.iloc[:, 1]
		label = df.iloc[:, 2]
		att_1_range = np.linspace(np.min(att_1), np.max(att_1), 100)
		
		plt.scatter(att_1, att_2, c = label, cmap = matplotlib.colors.ListedColormap(self.color))
		for CResult in CResults:
			#this is done to make the graph more visible
			if (i == 0 and CResult['C'] == 0.001):
				att_1_range = np.linspace(np.min(1.7), np.max(2.3), 100)
			elif (i == 0 and CResult['C'] == 0.01):
				att_1_range = np.linspace(np.min(2.27), np.max(2.35), 100)
			elif (i == 0 and (CResult['C'] == 0.1 or CResult['C'] == 1)): 
				att_1_range = np.linspace(np.min(att_1), np.max(att_1), 100)		
			att_2_val = (- CResult['intercept'] - att_1_range * CResult['coefficients'][0]) / CResult['coefficients'][1]
			plt.plot(att_1_range, att_2_val, label = 'C: ' + str(CResult['C']) + ' Score: ' + str(CResult['score']))
		plt.title(data['name'])
		plt.xlabel('Att 1')
		plt.ylabel('Att 2')
		plt.legend()
		plt.show()

	def run(self, i = 0):
		'''
		CResults = [{
			'C': 0.001 or 0.01 or 0.1 or 1,
			'score': accuracy,
			'coefficients': [coef[0], coef[1]],
			'intercept': y-intercept
		}]
		'''
		df_1 = self.readData(self.dataset_1)
		df_2 = self.readData(self.dataset_2)
		self.plotScatter(self.dataset_1, df_1)
		self.plotScatter(self.dataset_2, df_2)
		for df in [df_1, df_2]:
			CResults = []
			data = self.dataset_1 if (i == 0) else self.dataset_2 
			for C in self.C:
				score, coefficients, intercept = self.svm(df, C)
				CResult = {
					'C': C,
					'score': score,
					'coefficients': coefficients,
					'intercept': intercept
				}
				CResults.append(CResult)
			self.plotLine(i, data, df, CResults)
			i += 1
Q1()
