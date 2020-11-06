import numpy as np
import subprocess

n_samples = 100  # Discretization of a Trento parameter
n_trento = 4000  # Number of times to run Trento
wmin = 0.5
wmax = 1.2
xrange = np.arange(wmin, wmax, (wmax - wmin) / n_samples)
store = np.arange(n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
counti = 0

for i in xrange:
    string = '../build/src/trento Pb Pb ' + str(n_trento) + ' -w ' + str(i)
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
    ave = np.mean(data)
    print(i, ave)
    store[counti][0] = i
    store[counti][1] = ave
    counti += 1

np.save("datW", store)
