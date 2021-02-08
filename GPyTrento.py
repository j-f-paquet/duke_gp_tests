import numpy as np
import matplotlib.pyplot as plt
import GPy
import math
from IPython.display import display
from mpl_toolkits import mplot3d
import subprocess

import warnings

warnings.filterwarnings('ignore')
GPy.plotting.change_plotting_library('plotly_offline')

#### Trento Example more points ####
datum = np.load("datPW.txt.npy")
pw = np.array([[datum[i, 0], datum[i, 1]] for i in range(len(datum))])
e2 = np.array([[datum[i, 2]] for i in range(len(datum))])
e3 = np.array([[datum[i, 3]] for i in range(len(datum))])
print(pw)


def true2(tp, tw):
    string = '../build/src/trento Pb Pb 4000 -p ' + str(tp) + ' -w ' + str(tw)
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
    ave = np.mean(data)
    print(tp, tw, ave)
    return ave


def true3(tp, tw):
    string = '../build/src/trento Pb Pb 4000 -w ' + str(tw) + ' -p ' + str(tp)
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
    ave = np.mean(data)
    print(tp, tw, ave)
    return ave


truth1 = 0.314
truth2 = 0.618
min1 = 0
max1 = 0.5
min2 = 0.5
max2 = 1.2
# replace outputs with trento calls when actually running
output1 = true2(truth1, truth2)
output2 = true3(truth1, truth2)
fake_exp_rel_uncert_1 = 0.05
theory_rel_uncert_1 = 0.05
fake_exp_rel_uncert_2 = 0.05
theory_rel_uncert_2 = 0.05

cml1 = e2
cml2 = e3
cul1 = cml1 * theory_rel_uncert_1
cul2 = cml2 * theory_rel_uncert_2

# define kernel
ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)

# create simple GP model
mm = GPy.models.GPRegression(pw, e2, ker)

# optimize and plot
mm.optimize(messages=True, max_f_eval=1000)
fig = mm.plot()
display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
display(mm)


def predict(x, gp):
    return gp.predict(x)


def prior2():
    return 1


def likelihood2():
    res = 0.0
    norm = 1.
    # Sum over observables
    data_mean1 = output1
    data_uncert1 = output1 * fake_exp_rel_uncert_1

    tmp_model_mean1, tmp_model_uncert1 = predict(pw, mm)
    tmp_data_mean1 = data_mean1
    tmp_data_uncert1 = data_uncert1

    cov1 = (np.multiply(tmp_model_uncert1, tmp_model_uncert1) +
            np.multiply(tmp_data_uncert1, tmp_data_uncert1))
    res += np.divide(np.power(tmp_model_mean1 - tmp_data_mean1, 2), cov1)
    norm *= 1 / np.sqrt(cov1)

    # Again
    data_mean2 = output1
    data_uncert2 = output1 * fake_exp_rel_uncert_1

    tmp_model_mean2, tmp_model_uncert2 = predict(pw, mm)
    tmp_data_mean2 = data_mean2
    tmp_data_uncert2 = data_uncert2

    cov2 = (np.multiply(tmp_model_uncert2, tmp_model_uncert2) +
            np.multiply(tmp_data_uncert2, tmp_data_uncert2))
    res += np.divide(np.power(tmp_model_mean2 - tmp_data_mean2, 2), cov2)
    norm *= 1 / np.sqrt(cov2)

    res *= -0.5
    return norm * np.exp(res)


def posterior2():
    return prior2() * likelihood2()


ax = plt.axes(projection='3d')
ax.scatter(xs=pw[:, 0], ys=pw[:, 1], zs=posterior2())
ax.view_init(20, 10)

cs2 = plt.contourf(pw[:, 0].reshape(50, 70), pw[:, 1].reshape(50, 70), posterior2().reshape(50, 70), 20)

cbar2 = plt.colorbar(cs2, label="Posterior")

plt.plot(truth1, truth2, "D", color='red', ms=10)

plt.tight_layout()
plt.show()
