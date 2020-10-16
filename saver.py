import numpy as np
import subprocess

n_samples_p = 6   # Discretization of each Trento parameter
n_samples_sigma = 4   # Discretization of each Trento parameter
n_trento = 1000   # Number of times to run Trento
pmin = 0
pmax = 0.5
csmin = 0.5
csmax = 1.2
prange = np.arange(pmin, pmax, (pmax - pmin) / n_samples_p)
csrange = np.arange(csmin, csmax, (csmax - csmin) / n_samples_sigma)
store = np.arange(n_samples_p * n_samples_sigma * 4, dtype=np.float32).reshape(n_samples_p * n_samples_sigma, 4)
counti = 0

for i in prange:
    countj = 0
    for j in csrange:
        string = '../build/src/trento Pb Pb ' + str(n_trento) + ' -w ' + str(j) + ' -p ' + str(i)
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
        ave = np.mean(data)
        ave2 = np.mean(data2)
        print(i, j, ave, ave2)
        store[n_samples_sigma * counti + countj][0] = i
        store[n_samples_sigma * counti + countj][1] = j
        store[n_samples_sigma * counti + countj][2] = ave
        store[n_samples_sigma * counti + countj][3] = ave2
        countj += 1
    counti += 1

np.save("dat24.txt", store)
