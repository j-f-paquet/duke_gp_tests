# Load libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import subprocess

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

# suppression warning messages
import warnings

warnings.filterwarnings('ignore')

#################################
######## Define the model ######
#################################

# Replace these two functions with trento calls ################ <=============================
nlenp = 6
nlenx = 4
datum = np.load("dat24.txt.npy").reshape((nlenp, nlenx, 4))


# Return observable given parameter
def e2(params):
    thicc = params['reduced_thickness']
    nn = int(nlenp * thicc[0] * nlenx / 2)
    return datum[nn, :, 2]


def true2(params):
    thicc = params['reduced_thickness']
    xsec = params['cross_section']
    string = '../build/src/trento Pb Pb 4000 -p ' + str(thicc) + ' -x ' + str(xsec)
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 4]
    ave = np.mean(data)
    print(thicc, xsec, ave)
    return ave


# Return observable given parameter
def e3(params):
    thicc = params['reduced_thickness']
    nn = int(nlenp * thicc[0] * nlenx / 2)
    return datum[nn, :, 3]


def true3(params):
    thicc = params['reduced_thickness']
    xsec = params['cross_section']
    string = '../build/src/trento Pb Pb 4000 -x ' + str(xsec) + ' -p ' + str(thicc)
    with subprocess.Popen(string.split(), stdout=subprocess.PIPE) as proc:
        data = np.array([l.split() for l in proc.stdout], dtype=float)[:, 5]
    ave = np.mean(data)
    print(thicc, xsec, ave)
    return ave


# Dictionary of parameters
# "label" is used for plotting purposes
# "range" is the allowed range of the parameter (a simple uniform "prior")
# "truth" is the value of the parameters used for the closure test
parameter_d = {
    'reduced_thickness': {
        "label": "Reduced thickness",
        "range": [0, 0.5],  # <====================================================
        "truth": 0.314  # <====================================================
    },
    'cross_section': {
        "label": r"Cross-section $\sigma_{NN}$",
        "range": [4., 6.],  # <====================================================
        "truth": 5.28  # <====================================================
    }
}

# Observable dictionary
obs_d = {
    r"$\epsilon$2": {
        'fct': e2,
        'tfct': true2,
        'label': r"$\epsilon$2",
        'fake_exp_rel_uncert': 0.05,  # <====================================================
        'theoretical_relative_uncertainty': 0.05  # <====================================================
    },
    r"$\epsilon$3": {
        'fct': e3,
        'tfct': true3,
        'label': r"$\epsilon$3",
        'fake_exp_rel_uncert': 0.05,  # <====================================================
        'theoretical_relative_uncertainty': 0.05  # <====================================================
    }
}

#########################
#### Get the "data" #####
#########################

# Data dictionary
data_d = {}

# This will be a "closure test": we use the model to generate "data"
# The Bayesian parameter estimation should be peaked around the true value of the parameters
for obs_name, info_d in obs_d.items():
    obs_fct = info_d['tfct']
    data_d[obs_name] = {}
    param_value_list = {item: tmp_d['truth'] for (item, tmp_d) in parameter_d.items()}
    tmp_value = obs_fct(param_value_list)
    data_d[obs_name]['mean'] = tmp_value
    fake_exp_rel_uncert = info_d['fake_exp_rel_uncert']
    data_d[obs_name]['uncert'] = fake_exp_rel_uncert * tmp_value

#########################
#### Plot the "data" #####
#########################

for obs_name, info_d in obs_d.items():
    # Function that returns the value of an observable
    obs_fct = info_d['fct']

    # Label for the observable
    obs_label = info_d['label']

    # Data
    tmp_data_d = data_d[obs_name]
    data_mean = tmp_data_d['mean']
    data_uncert = tmp_data_d['uncert']

    # Info about parameters
    param_name_list = list(parameter_d.keys())

    x_param_name = param_name_list[0]
    xmin, xmax = parameter_d[x_param_name]['range']
    x_label = parameter_d[x_param_name]['label']

    y_param_name = param_name_list[1]
    ymin, ymax = parameter_d[y_param_name]['range']
    y_label = parameter_d[y_param_name]['label']

    # Plot what the observable looks like over the parameter prior
    plt.figure()
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Compute the posterior for a range of values of the parameter "x"
    x_range = np.arange(xmin, xmax, (xmax - xmin) / nlenp)
    y_range = np.arange(ymin, ymax, (ymax - ymin) / nlenx)

    x_mesh, y_mesh = np.meshgrid(x_range, y_range, sparse=False, indexing='ij')
    print(np.shape(x_mesh), np.shape(y_mesh))

    z_list = [obs_fct({x_param_name: x_val, y_param_name: y_val}) for (x_val, y_val) in zip(x_mesh, y_mesh)]
    print(np.shape(z_list))
    # Plot the posterior
    cs = plt.contourf(x_mesh, y_mesh, z_list, 20)
    cbar = plt.colorbar(cs, label=obs_label)

    # Plot the "data"
    # plt.contour(x_mesh, y_mesh, z_list, levels=[data_mean], colors='r')
    # plt.contourf(x_mesh, y_mesh, z_list, levels=[np.array(data_mean) - np.array(data_uncert),
    #                                             np.array(data_mean) + np.array(data_uncert)], colors='r', alpha=.4)
    plt.tight_layout()
    plt.show()

########################
# Get the calculations #
########################

calc_d = {}

for obs_name, info_d in obs_d.items():
    # Number of points used for the "emulator"
    number_design_emulator = 3  # <=========================================================

    # Function that returns the value of an observable
    obs_fct = info_d['fct']

    # Info about parameters
    param_name_list = list(parameter_d.keys())

    x_param_name = param_name_list[0]
    xmin, xmax = parameter_d[x_param_name]['range']

    y_param_name = param_name_list[1]
    ymin, ymax = parameter_d[y_param_name]['range']

    # For simplicity, we sample the emulator uniformly
    x_list = np.linspace(xmin, xmax-(xmax-xmin)/nlenp, num=number_design_emulator)
    y_list = np.linspace(ymin, ymax-(ymax-ymin)/nlenx, num=number_design_emulator)

    x_mesh2, y_mesh2 = np.meshgrid(x_list, y_list, sparse=False, indexing='ij')

    calculation_mean_list = [obs_fct({x_param_name: x_val, y_param_name: y_val})
                             for (x_val, y_val) in zip(x_mesh2, y_mesh2)]

    relative_uncertainty = info_d['theoretical_relative_uncertainty']

    calculation_uncert_list = np.multiply(calculation_mean_list, relative_uncertainty)

    calc_d[obs_name] = {'x_list': x_list, 'y_list': y_list, 'mean': calculation_mean_list,
                        'uncert': calculation_uncert_list}

#########################################
# Make interpolator for each observable #
#########################################

kernel = (
        1. * kernels.RBF(
    length_scale=.2,
    length_scale_bounds=(.05, .5)
)
        #    + kernels.ConstantKernel()
        + kernels.WhiteKernel(
    noise_level=1.,
    noise_level_bounds=(1e-5, 1e5))
)

gp = GPR(kernel=kernel,
         n_restarts_optimizer=5,
         copy_X_train=False)
meshmesh = np.zeros((nlenp * nlenx, 2))
for ii in range(nlenp * nlenx):
    meshmesh[ii][0] = x_mesh[int((ii*nlenp) % (nlenp*nlenx))][0]
    meshmesh[ii][1] = y_mesh[0][int(ii % nlenx)]
print(meshmesh)
gp.fit(np.atleast_2d(meshmesh), z_list)
print("C^2 = ", gp.kernel_.get_params()['k1'])
print(gp.kernel_.get_params()['k2'])


def predictM(x, gpx):
    mean2, cov = gpx.predict(return_cov=True, X=np.atleast_2d(x).T)
    print(mean2)
    return mean2


def predictC(x, gpx):
    mean2, cov = gpx.predict(return_cov=True, X=np.atleast_2d(x).T)
    return np.sqrt(np.diag(cov))


emul_d = {}

for obs_name, info_d in obs_d.items():
    emul_d[obs_name] = {
        'mean': predictM(x_mesh, gp),
        'uncert': predictC(x_mesh, gp)
    }


#########################
# Compute the posterior #
#########################

# We assume uniform priors for this example
# Here 'x' is the only model parameter
def prior():
    return 1


# Under the approximations that we're using, the posterior is exp(-1/2*\sum_{observables, pT} (model(observable,
# pT)-data(observable,pT))^2/(model_err(observable,pT)^2+exp_err(observable,pT)^2)

# Here 'x' is the only model parameter
def likelihood(params, data):
    res = 0.0
    norm = 1.
    # Sum over observables
    for obs_name2 in obs_d.keys():
        emul_calc = emul_d[obs_name2]['mean']
        emul_uncert = emul_d[obs_name2]['uncert']
        data_mean2 = data[obs_name2]['mean']
        data_uncert2 = data[obs_name2]['uncert']
        emul_calc_vec = np.vectorize(emul_calc)
        emul_uncert_vec = np.vectorize(emul_uncert)
        param_name_list2 = list(parameter_d.keys())
        x_param_name2 = param_name_list2[0]
        x_value = params[x_param_name2]
        y_param_name2 = param_name_list2[1]
        y_value = params[y_param_name2]

        tmp_model_mean = emul_d[obs_name2]['mean']
        tmp_model_uncert = emul_d[obs_name2]['uncert']
        tmp_data_mean = data_mean2
        tmp_data_uncert = data_uncert2
        cov = (np.multiply(tmp_model_uncert, tmp_model_uncert) + np.multiply(tmp_data_uncert, tmp_data_uncert))
        res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov
        norm *= 1 / np.sqrt(cov)
    res *= -0.5
    return norm * np.exp(res)


def posterior(params, data):
    return prior() * likelihood(params, data)


##################
# Plot posterior #
##################


# Info about parameters
param_name_list = list(parameter_d.keys())

x_param_name = param_name_list[0]
xmin, xmax = parameter_d[x_param_name]['range']
x_truth = parameter_d[x_param_name]['truth']
x_label = parameter_d[x_param_name]['label']

y_param_name = param_name_list[1]
ymin, ymax = parameter_d[y_param_name]['range']
y_truth = parameter_d[y_param_name]['truth']
y_label = parameter_d[y_param_name]['label']

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(x_label)
plt.ylabel(y_label)

# Compute the posterior for a range of values of the parameter "x"
x_range = np.arange(xmin, xmax, (xmax - xmin) / 100.)
y_range = np.arange(ymin, ymax, (ymax - ymin) / 100.)

x_mesh3, y_mesh3 = np.meshgrid(x_range, y_range, sparse=False, indexing='ij')

posterior_array = [posterior({x_param_name: x_val, y_param_name: y_val}, data_d)
                   for (x_val, y_val) in zip(x_mesh3, y_mesh3)]
print(np.shape(posterior_array))
# Plot the posterior
cs2 = plt.contourf(x_mesh3, y_mesh3, posterior_array, 20)

cbar2 = plt.colorbar(cs2, label="Posterior")

plt.plot([x_truth], [y_truth], "D", color='red', ms=10)

plt.tight_layout()
plt.show()

###############################
# Plotting marginal posterior #
###############################

# Posterior vs x
plt.figure()
plt.xscale('linear')
plt.yscale('linear')
# plt.xlim(0,2)
# plt.ylim(1e-5,1e2)
plt.xlabel(x_label)
plt.ylabel(r'Posterior')

# The marginal posterior for a parameter is obtained by integrating over a subset of other model parameters

# Compute the posterior for a range of values of the parameter "x"
x_range = np.arange(xmin, xmax, (xmax - xmin) / 100.)

posterior_list = [
    scipy.integrate.quad(lambda y_val: posterior({x_param_name: x_val, y_param_name: y_val}, data_d), ymin, ymax)[0] for
    x_val in x_range]

plt.plot(x_range, posterior_list, "-", color='black', lw=4)

plt.axvline(x=x_truth, color='red')

plt.tight_layout()
# plt.show()


# Posterior vs y
plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(y_label)
plt.ylabel(r'Posterior')

# Compute the posterior for a range of values of the parameter "x"
y_range = np.arange(ymin, ymax, (ymax - ymin) / 100.)

posterior_list = [
    scipy.integrate.quad(lambda x_val: posterior({x_param_name: x_val, y_param_name: y_val}, data_d), xmin, xmax,
                         limit=100, epsrel=1e-4)[0] for y_val in y_range]

plt.plot(y_range, posterior_list, "-", color='black', lw=4)

plt.axvline(x=y_truth, color='red')

plt.tight_layout()
plt.show()
