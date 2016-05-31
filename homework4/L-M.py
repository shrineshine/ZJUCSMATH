import numpy as np

#fitting data, from 'Mathematical Experiment'
x = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
y = [19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01]

#assumption
x0 = 0.5
y0 = 10
y_init = y0 * np.exp([-x0 * w for w in x])

num = len(x)
dim = 2
iterationMax = 20
#initial lamda for L-M
lamda = 0.01

#assignment
update = 1

#iteration
for i in range(iterationMax):
	if update == 1:
		#calculate Jacobi matrix
		JacobiM = np.zeros(num * dim).reshape(num, dim)
		for j in range(num):
			JacobiM[j, :] = [np.exp(-x0 * x[j]), -y0 * x[j] * np.exp(-x0 * x[j])]
		#calculate new y(y_fit)
		y_fit = y0 * np.exp([-x0 * w for w in x])
		#calculate distance
		dis = y - y_fit
		#Hessian matrix
		H = np.dot(JacobiM.T, JacobiM)
		#calculate deviation
		if i==0:
			deviation = np.dot(dis, dis)
	H_lm = H + (lamda * np.identity(dim))
	#calculate step length
	step = np.dot(np.mat(H_lm).I, np.dot(JacobiM.T, dis[:]))
	g = np.dot(JacobiM.T, dis[:])
	#try to move x0, y0
	y_lm = y0 + step[0, 0]
	x_lm = x0 + step[0, 1]
	y_fit_lm = y_lm * np.exp([-x_lm * w for w in x])
	#update deviation
	dis_lm = y - y_fit_lm
	deviation_lm = np.dot(dis_lm, dis_lm)

	#update parameters and lamda depending on deviation_lm
	if deviation_lm < deviation:
		lamda = lamda / 10
		y0 = y_lm
		x0 = x_lm
		deviation = deviation_lm
		update = 1
	else:
		lamda = lamda * 10
		update = 0
	print "iter: " + str(i) +"  deviation: " + str(deviation) + "  lamda: " + str(lamda)
print "y0 = " + str(y0)
print "x0 = " + str(x0)
