import numpy as np
import matplotlib.pyplot as plt

#using simplified sequential minimal optimization
def smo(dataMatrix, label, C, tolerate, iterationMax):
	iterator = 0
	b = 0
	num, dim = np.shape(dataMatrix)
	alphas = np.mat(np.zeros((num, 1)))
	while iterator < iterationMax:
		#initialize alphaPairsChanged which stores whether alpha has been optimized
		alphaPairsChanged = 0
		for i in range(num):
			#the class we anticipate
			label_i = float(np.multiply(alphas, label).T * (dataMatrix * dataMatrix[i, :].T)) + b
			#calculate deviation
			deviation_i = label_i - float(label[i])
			#if deviation is too big, start optimization
			if((label[i] * deviation_i < -tolerate) and (alphas[i] < C) or (label[i] * deviation_i > tolerate) and (alphas[i] > 0)):
				#select a random alpha index
				r = i
				while r == i:
					r = int(np.random.uniform(0, num))
				#calculate r's label and deviation
				label_r = float(np.multiply(alphas, label).T * (dataMatrix * dataMatrix[r, :].T)) + b
				deviation_r = label_r - float(label[r])
				alpha_i_old = alphas[i].copy()
				alpha_r_old = alphas[r].copy()
				#ensure alpha[r] is between 0 and C
				if label[i] != label[r]:
					L = max(0, alphas[r] - alphas[i])
					H = min(C, C + alphas[r] - alphas[i])
				else:
					L = max(0, alphas[r] + alphas[i] - C)
					H = min(C, alphas[r] + alphas[i])
				if L == H:
					continue
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[r, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[r, :] * dataMatrix[r, :].T
				if eta >= 0:
					continue
				alphas[r] -= label[r] * (deviation_i - deviation_r) / eta
				#adjust alpha if alpha is bigger than H or less than L
				if alphas[r] > H:
					alphas[r] = H
				if L > alphas[r]:
					alphas[r] = L
				if abs(alphas[r] - alpha_r_old) < 0.00001:
					continue
				#modify i with same step but opposite direction compared to r
				alphas[i] += label[r] * label[i] * (alpha_r_old - alphas[r])
				b1 = b - deviation_i - label[i] * (alphas[i] - alpha_i_old) * dataMatrix[i, :] * dataMatrix[i, :].T - label[r] * (alphas[r] - alpha_r_old) * dataMatrix[i, :] * dataMatrix[r, :].T
				b2 = b - deviation_r - label[i] * (alphas[i] - alpha_i_old) * dataMatrix[i, :] * dataMatrix[r, :].T - label[r] * (alphas[r] - alpha_r_old) * dataMatrix[r, :] * dataMatrix[r, :].T
				if (0 < alphas[i]) and (C > alphas[i]):
					b = b1
				elif (0 < alphas[r]) and (C > alphas[r]):
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alphaPairsChanged = alphaPairsChanged + 1
		#only stop when no changes habben on alpha
		if alphaPairsChanged == 0:
			iterator = iterator + 1
		else:
			iterator = 0
	return b, alphas
#read data set, construct dataSet and label
dataSet = []
label = []
for line in open("DataSet.txt"):
	line = line.rstrip().split(" ")
	dataSet.append([float(line[0]), float(line[1])])
	label.append(float(line[2]))
dataMatrix = np.mat(dataSet)
label = np.mat(label).transpose()
num, dim = np.shape(dataMatrix)
b, alphas = smo(dataMatrix, label, 0.6, 0.01, 100)

plt.axis([0, 10, 0, 10])
for i in range(num):
	if label[i] == -1:
		plt.plot(dataMatrix[i, 0], dataMatrix[i, 1], 'r+')
	elif label[i] == 1:
		plt.plot(dataMatrix[i, 0], dataMatrix[i, 1], 'ob') 
supportVectorsIndex = np.nonzero(alphas > 0)[0]  
for i in supportVectorsIndex:
	plt.plot(dataMatrix[i, 0], dataMatrix[i, 1], 'oy') 
w = np.zeros((2, 1))
for i in supportVectorsIndex:
	w += np.multiply(alphas[i] * label[i], dataMatrix[i, :].T)
margin = 2/np.sqrt(np.dot(w[1:3],w[1:3]))
min_x = min(dataMatrix[:, 0])[0, 0]
max_x = max(dataMatrix[:, 0])[0, 0]
y_min_x = float(-b - w[0] * min_x) / w[1]
y_max_x = float(-b - w[0] * max_x) / w[1]
plt.plot([min_x, max_x], [y_min_x, y_max_x], '-r', label = "Decision Boundary")
plt.legend()
plt.show()  
