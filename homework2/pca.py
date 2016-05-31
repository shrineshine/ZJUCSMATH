from pca_module import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
numVec = []
numVecMatrix = []
#Open file
fileHandle = open ('optdigits-orig.tra') 
fileList = fileHandle.readlines()
lineNum = 0
flag = False
#Get all vectors whose number is 3
for fileLine in fileList:
	lineNum = lineNum + 1
	line = fileLine.rstrip()
	if lineNum < 22:
		continue
	if (lineNum - 21) % 33 != 0 :
		numVec += [int(x) for x in line]
	else:
		#If number is 3, store the vector. Then clear the numVec
		if(line == ' 3'):
			numVecMatrix.append(numVec)
		numVec = []
fileHandle.close()

#Minus mean
matrix = np.array(numVecMatrix)
meanVals = np.mean(matrix, axis=0)  
metrixAve = matrix - meanVals

#Calculate covariance matrix
cov = np.cov(metrixAve, rowvar=0) 
evals,evecs = np.linalg.eig(np.mat(cov)) 
indices = np.argsort(evals)
indices = indices[-1:-3:-1]
evecs = evecs[:,indices]
#Projection
lowDDataMat = metrixAve * evecs 

#Plot
dataArr1 = np.array(lowDDataMat)
m = np.shape(dataArr1)[0]
axis_x1 = []
axis_y1 = []
for i in range(m):
	axis_x1.append(dataArr1[i,0])
	axis_y1.append(dataArr1[i,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
# plt.axis([-7, 8, -10, 10])
plt.plot(axis_x1,axis_y1,"o")
plt.show()