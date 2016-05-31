import matplotlib.pyplot as plt
import numpy as np

def Gaussian2D(mean, var, num):
	points = [];
	points.append(mean[0] + np.random.randn(num) * var[0])
	points.append(mean[1] + np.random.randn(num) * var[1])
	return points
def initial(classes, dim):
	miu = np.zeros((classes, dim))
	Sigma = np.zeros((classes, dim, dim))
	detSigma = np.zeros(classes)
	invSigma = Sigma.copy()
	Pi = np.zeros(classes)
	for cl in range(classes):
		Pi[cl] = np.random.rand()
		miu[cl] = np.random.rand(dim)
		Sigma[cl] = np.eye(dim, dim)
		invSigma[cl] = np.linalg.inv(Sigma[cl])
		detSigma[cl] = np.linalg.det(Sigma[cl])
		Pi=Pi/sum(Pi) 
	return (miu, Sigma, detSigma, invSigma, Pi)
samples1 = Gaussian2D([-3,-4], [1,1], 100)
samples2 = Gaussian2D([2,5], [1,1], 100)
samples = np.vstack([np.hstack([samples1[0], samples2[0]]) , np.hstack([samples1[1], samples2[1]])])
samples = samples.T
classes = 2
rows, dim = samples.shape
#initial values
miu, Sigma, detSigma, invSigma, Pi = initial(classes, dim)
gamma=np.zeros((rows,dim))
iterator = 0
while iterator < 200:
	iterator += 1
	for row in range(rows):
		for cl in range(classes):
			#calculate the coefficient of MOG's exponent
			coef = (2 * np.pi) ** (-len(samples[row]) * 1.0 / 2) * (detSigma[cl] ** (-0.5))
			#calculete the probability of one Gaussian distribution
			vsum = coef * np.exp(-0.5 * np.dot(np.dot(samples[row] - miu[cl], invSigma[cl]), samples[row] - miu[cl]))
			gamma[row, cl] = Pi[cl] * vsum
		#calculate every sample's probibility generated by each Gaussian distribution 
		gamma[row] = gamma[row] / sum(gamma[row])
	Nk = sum(gamma, 0)
	#update Pi
	Pi = Nk / rows
	miu = np.zeros((classes, dim))
	Sigma = np.zeros((classes, dim, dim))
	for cl in range(classes):
		#recalculate the miu of each component
		for row in range(rows):
			miu[cl] += gamma[row][cl]*samples[row]
		miu[cl] = miu[cl] / Nk[cl]
		#recalculate the cov of each component
		for row in range(rows):
			vsum = np.mat(samples[row] - miu[cl])
			Sigma[cl] += gamma[row][cl] * np.dot(vsum.T, vsum)
        Sigma[cl] = Sigma[cl]/Nk[cl]
        detSigma[cl] = np.linalg.det(Sigma[cl])
        invSigma[cl] = np.linalg.inv(Sigma[cl])  
idx = np.zeros(rows)
for row in range(rows):
	idx[row] = np.nonzero(gamma[row] == max(gamma[row]))[0][0]
for cl in range(classes):
	t = np.nonzero(idx == cl)[0]
	plt.plot(samples[t, 0], samples[t, 1], 'o', label = 'Group ' + str(row + 1))
plt.show()
