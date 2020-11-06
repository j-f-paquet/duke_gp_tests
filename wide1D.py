########## Block 1 ############## <-- Please refer this block number when you ask questions
# suppression warning messages
import warnings

# package for plotting
import matplotlib.pyplot as plt
import numpy as np
# scikit-learn: machine learning in Python
# https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# In this example, we are using the principal component analysis
# and the Gaussian process regression as implemented in sklearn.
# Detailed documentation of these modules:
# Principal component analysis (PCA):
# https://scikit-learn.org/stable/modules/decomposition.html
# Gaussian process gressor (GPR)):
# https://scikit-learn.org/stable/modules/gaussian_process.html
from sklearn.gaussian_process import \
    GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

warnings.filterwarnings('ignore')

# some plot settings, not important
fontsize = 12
plt.rcdefaults()
plt.rcParams.update({
    'font.size': fontsize,
    'legend.fontsize': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': True,
    'axes.spines.right': True,
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
})

xmin = 0.5
xmax = 1.2
dataW = np.load("datW.npy")
step = (xmax - xmin) / len(dataW)


def observable_1(x):
    nn = int((x - xmin) / step)
    return dataW[nn, 1]


x_truth = 0.88

# This will be a "closure test": we use the model to generate "data"
# The Bayesian parameter estimation should be peaked around "x_truth"
data_mean = observable_1(x_truth)

# Real data come with uncertainties
# Here, we just define an arbitrary uncertainty on the "data"
relative_uncertainty = .05  # <=========================================================
data_uncert = relative_uncertainty * data_mean

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(r'$x$')
plt.ylabel(r'observable_1')

# Compute the posterior for a range of values of the parameter "x"
x_range = np.arange(xmin, xmax, (xmax - xmin) / len(dataW))
y_list = [observable_1(x) for x in x_range]

# Plot the posterior
plt.plot(x_range, y_list, "-", color='black', lw=2)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Plot the true value of the parameter "x", for comparison
# plt.axvline(x=x_truth, color='red')

# Plot the "data"
# plt.axhline(y=data_mean)
# plt.fill_between(x_range, data_mean-data_uncert, data_mean+data_uncert)

plt.tight_layout()
# plt.show()

########################
# Get the calculations #
########################

# Number of points used for the "emulator"
number_design_emulator = 10  # <=========================================================

# For simplicity, we sample the emulator uniformly
x_design = np.linspace(xmin, xmax - step, num=number_design_emulator)
y_design = [observable_1(x) for x in x_design]

# plot the design points, along with the full function
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(x_design, y_design, 'ro', label='Design')
x = np.linspace(xmin, xmax, 2001)

kernel = (1. * kernels.RBF(
          length_scale=.2,
          length_scale_bounds=(.05, .5)
          )
          # + kernels.ConstantKernel()
          + kernels.WhiteKernel(
            noise_level=1.,
            noise_level_bounds=(1e-5, 1e5))
          )

gp = GPR(kernel=kernel,
         n_restarts_optimizer=5,
         copy_X_train=False)
gp.fit(np.atleast_2d(x_design).T, y_design)
print("C^2 = ", gp.kernel_.get_params()['k1'])
print(gp.kernel_.get_params()['k2'])


def predict(x, gpx):
    mean, cov = gpx.predict(return_cov=True, X=np.atleast_2d(x).T)
    return mean, np.sqrt(np.diag(cov))


xArange = np.linspace(xmin, xmax, 201)
y, ystd = predict(xArange, gp)
figNew, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(x_design, y_design, 'ro', label='Design')
# ax.plot(x, [observable_1(d) for d in x],'k-', label=r'$F(x)$')
ax.plot(x_design, y_design, 'k-', label=r'$F(x)$')
ax.plot(xArange, y, 'b--', label=r'GP mean')
ax.fill_between(xArange, y - ystd, y + ystd, color='b', alpha=.3, label=r'GP $\pm 1\sigma$')
ax.fill_between(xArange, y - 2 * ystd, y + 2 * ystd, color='gray', alpha=.3, label=r'GP $\pm 2\sigma$')
ax.set_xlabel(r"Input $x$")
ax.set_ylabel(r"Output $y=F(x)$")
ax.legend()


#########################
# Compute the posterior #
#########################

# We assume uniform priors for this example
# Here 'x' is the only model parameter
def prior():
    return 1


# Under the approximations that we're using, the likelihood is exp(-1/2*\sum_{observables, pT} (model(observable,
# pT)-data(observable,pT))^2/(model_err(observable,pT)^2+exp_err(observable,pT)^2)

# Here 'x' is the only model parameter
def likelihood(x):
    res = 0.0
    tmp_model_mean, tmp_model_uncert = predict(x, gp)
    tmp_data_mean = data_mean
    tmp_data_uncert = data_uncert
    cov = (tmp_model_uncert * tmp_model_uncert + tmp_data_uncert * tmp_data_uncert)
    res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov
    res *= -0.5
    return np.exp(res) / np.sqrt(cov)


#
def posterior(x):
    return prior() * likelihood(x)


##################
# Plot posterior #
##################

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
# plt.xlim(0,2)
# plt.ylim(1e-5,1e2)
plt.xlabel(r'$x$')
plt.ylabel(r'Posterior')

# Compute the posterior for a range of values of the parameter "x"
x_range = np.arange(xmin, xmax, (xmax - xmin) / 1000.)
posterior_list = [posterior(x) for x in x_range]

# Plot the posterior
plt.plot(x_range, posterior_list, "-", color='black', lw=4)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Plot the true value, for comparison
plt.axvline(x=x_truth, color='red')

plt.tight_layout()
plt.show()
