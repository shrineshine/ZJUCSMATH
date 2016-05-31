import numpy as np
output = open('DataSet.txt', 'w')
x = np.random.random(100) * 10
y = np.random.random(100) * 10
delnum = 1
for i in range(len(x)):
	if ( x[i] * 1.63 - 3.15 - y[i] ) > 0:
		if ( x[i] * 1.63 - 3.15 - y[i] ) < delnum:
			continue
		output.write(str(x[i]) + " " + str(y[i]) + " 1\n")
	else:
		if ( x[i] * 1.63 - 3.15 - y[i] ) > -delnum:
			continue
		output.write(str(x[i]) + " " + str(y[i]) + " -1\n")
