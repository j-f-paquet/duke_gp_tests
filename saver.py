import numpy as np
import subprocess

n = 100
m = 1000
pmin = 0
pmax = 2
csmin = 4
csmax = 10
prange = np.arange(pmin, pmax, (pmax - pmin) / n)
csrange = np.arange(csmin, csmax, (csmax - csmin) / n)
store = np.arange(n * n * 4, dtype=np.float32).reshape(n * n, 4)
counti = 0

for i in prange:
    countj = 0
    for j in csrange:
        string = '../build/src/trento Pb Pb ' + str(m) + ' -x ' + str(j) + ' -p ' + str(i)
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
        ave = np.mean(data)
        ave2 = np.mean(data2)
        print(i, j, ave, ave2)
        store[n * counti + countj][0] = i
        store[n * counti + countj][1] = j
        store[n * counti + countj][2] = ave
        store[n * counti + countj][3] = ave2
        countj += 1
    counti += 1

np.save("dat.txt", store)
