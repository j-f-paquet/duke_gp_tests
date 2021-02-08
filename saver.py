import numpy as np
import subprocess

accessFileName = "paramsBig"
dataFileName = "datGeneralBig"
numParams = 2
nSamplesParam1 = 50   # Discretization of each Trento parameter
nSamplesParam2 = 70   # Discretization of each Trento parameter
nTrentoRuns = 4000   # Number of times to run Trento
param1min = 0
param1max = 0.5
param1name = "Reduced thickness"
param1truth = 0.314
param2min = 0.5
param2max = 1.2
param2name = "Nucleon-Width"
param2truth = 0.618
numObs = 2
obs1name = r"$\epsilon$2"
obs2name = r"$\epsilon$3"

string = '../build/src/trento Pb Pb ' + str(nTrentoRuns) + ' -p ' + str(param1truth) + ' -w ' + str(param2truth)
with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
    datum = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
    datum2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
aver = np.mean(datum)
aver2 = np.mean(datum2)
print(param1truth, param2truth, aver, aver2)

# Storage: data file name, number of parameters, [parameter sizes], [parameter names],
#          [parameter min values], [parameter max values], [parameter truths],
#          number of observables, [observable names], [observable truths]
store1 = np.array([dataFileName, numParams, [nSamplesParam1, nSamplesParam2], [param1name, param2name],
                   [param1min, param2min], [param1max, param2max], [param1truth, param2truth],
                   numObs, [obs1name, obs2name], [aver, aver2]], dtype=object)

np.save(accessFileName, store1)
print("Saved parameters file")

param1range = np.arange(param1min, param1max, (param1max - param1min) / nSamplesParam1)
param2range = np.arange(param2min, param2max, (param2max - param2min) / nSamplesParam2)
store2 = np.arange(nSamplesParam1 * nSamplesParam2 * 4, dtype=np.float32).reshape(nSamplesParam1 * nSamplesParam2, 4)
counti = 0

for i in param1range:
    countj = 0
    for j in param2range:
        string = '../build/src/trento Pb Pb ' + str(nTrentoRuns) + ' -w ' + str(j) + ' -p ' + str(i)
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
        with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
            data2 = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
        ave = np.mean(data)
        ave2 = np.mean(data2)
        print(i, j, ave, ave2)
        store2[nSamplesParam2 * counti + countj][0] = i
        store2[nSamplesParam2 * counti + countj][1] = j
        store2[nSamplesParam2 * counti + countj][2] = ave
        store2[nSamplesParam2 * counti + countj][3] = ave2
        countj += 1
    counti += 1

np.save(dataFileName, store2)
