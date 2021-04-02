# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Use Gaussian process from scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels

# suppression warning messages
import warnings

warnings.filterwarnings('ignore')

# Storage: data file name, [parameter sizes], [parameter names], [parameter min values],
#          [parameter max values], [parameter truths], [observable names], [observable truths],
#          [experimental relative uncertainty], [theoretical relative uncertainty]
savedValues = np.load("listedSmall.npy", allow_pickle=True)
paramSizes = savedValues[1]
paramNames = savedValues[2]
paramMins = savedValues[3]
paramMaxs = savedValues[4]
paramTruths = savedValues[5]
obsNames = savedValues[6]
obsTruths = savedValues[7]
expRelUncert = savedValues[8]
theoRelUncert = savedValues[9]

# datum: np.array([[design_points], [observables]])
datum = np.load(str(savedValues[0]) + ".npy", allow_pickle=True)
desPts = datum[0]
observables = datum[1]

### Add uncertainty to the observables ###
calcUncertList = np.multiply(observables, theoRelUncert)
noise = np.zeros(np.shape(calcUncertList))
for ii in range(len(calcUncertList)):
    for jj in range(len(obsTruths)):
        noise[ii][jj] = np.random.normal(0, calcUncertList[ii][jj])
calcMeanPlusNoise = np.add(observables, noise)

### Make emulator for each observable ###
emul_d = {}

for nn in range(len(obsTruths)):
    # Label for the observable
    obs_label = obsNames[nn]

    # Function that returns the value of an observable (just to get the truth)
    param1_label = paramNames[0]
    param1_nb_design_pts = desPts[:, 0]
    param1_truth = paramTruths[0]

    param2_label = paramNames[0]
    param2_nb_design_pts = desPts[:, 1]
    param2_truth = paramTruths[0]

    param1_paramspace_length = paramMaxs[0] - paramMins[0]
    param2_paramspace_length = paramMaxs[1] - paramMins[1]

    # Kernels
    k0 = 1. * kernels.RBF(
        length_scale=(param1_paramspace_length / 2., param2_paramspace_length / 2.),
        length_scale_bounds=(
            (param1_paramspace_length / param1_nb_design_pts, 3. * param1_paramspace_length),
            (param2_paramspace_length / param2_nb_design_pts, 3. * param2_paramspace_length)
        )
    )

    k2 = 1. * kernels.WhiteKernel(
        noise_level=theoRelUncert[nn],
        # noise_level_bounds='fixed'
        noise_level_bounds=(theoRelUncert[nn] / 4., 4 * theoRelUncert[nn])
    )

    kernel = (k0 + k2)

    nrestarts = 10

    emulator_design_pts_value = desPts.tolist()

    emulator_obs_mean_value = observables[:, nn].tolist()

    # Fit a GP (optimize the kernel hyperparameters) to each PC.
    gaussian_process = GPR(
        kernel=kernel,
        alpha=0.0001,
        n_restarts_optimizer=nrestarts,
        copy_X_train=True
    ).fit(emulator_design_pts_value, emulator_obs_mean_value)

    # https://github.com/keweiyao/JETSCAPE2020-TRENTO-BAYES/blob/master/trento-bayes.ipynb
    print('Information on emulator for observable ' + obs_label)
    print('RBF: ', gaussian_process.kernel_.get_params()['k1'])
    print('White: ', gaussian_process.kernel_.get_params()['k2'])

    emul_d[obsNames[nn]] = {
        'gpr': gaussian_process
        # 'mean':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'], np.transpose(
        # calc_d[obs_name]['mean']), kind='linear', copy=True, bounds_error=False, fill_value=None),
        # 'uncert':scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'], np.transpose(
        # calc_d[obs_name]['uncert']), kind='linear', copy=True, bounds_error=False, fill_value=None)
    }

    #####################
    # Plot the emulator #
    #####################

    # observable vs value of one parameter (with the other parameter fixed)
    plt.figure()
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(param1_label)
    plt.ylabel(obs_label)

    len_y = paramSizes[1]
    step_y = np.max([int(len_y / 2), 1])

    for iy in np.arange(0, len_y, step_y):
        y = desPts[iy][1]
        y_label = param2_label

        # Compute the posterior for a range of values of the parameter "x"
        x_range = np.linspace(paramMins[0], paramMins[1], 50)
        y_range = np.full_like(x_range, y)

        param_value_array = np.transpose([x_range, y_range])

        z_list, z_list_uncert = gaussian_process.predict(param_value_array, return_std=True)
        # print('param_value_array',param_value_array)
        # print('z_list',z_list)

        # print('all=',calc_d[obs_name]['mean'])
        # print('shape=',np.array(calc_d[obs_name]['mean']).shape)
        # print('w iy',np.array(calc_d[obs_name]['mean'])[:,iy])

        # Plot design points
        plt.errorbar(desPts[:, 0], np.array(observables[nn])[:, iy],
                     yerr=np.array(calcUncertList)[:, iy], fmt='D', color='orange', capsize=4,
                     label="" + y_label + "=" + str(y))

        # print(calc_d[obs_name]['x_list'],calc_d[obs_name]['mean'][iy],calc_d[obs_name]['uncert'][iy])

        # Plot interpolator
        plt.plot(x_range, z_list, color='blue')
        plt.fill_between(x_range, z_list - z_list_uncert, z_list + z_list_uncert, color='blue', alpha=.4)

    # Plot the truth
    plt.plot(param1_truth, obsTruths[nn], "D", color='black')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()


### Compute the Posterior ###
# We assume uniform priors for this example
# Here 'x' is the only model parameter
def prior():
    return 1


# Under the approximations that we're using, the posterior is
# exp(-1/2*\sum_{observables, pT}
# (model(observable,pT)-data(observable,pT))^2/(model_err(observable,pT)^2+exp_err(observable,pT)^2)

# Here 'x' is the only model parameter
def likelihood():
    res = 0.0

    norm = 1.

    # Sum over observables
    for xx in range(len(obsTruths)):
        # Function that returns the value of an observable

        # emulator_calc=emul_d[obs_name]['mean']
        # emulator_uncert=emul_d[obs_name]['uncert']

        data_mean2 = observables[:, xx]
        data_uncert2 = np.multiply(data_mean2, expRelUncert[xx])

        tmp_data_mean, tmp_data_uncert = data_mean2, data_uncert2

        tmp_model_mean, tmp_model_uncert = gaussian_process.predict(
            np.atleast_2d(desPts), return_std=True)

        cov = (tmp_model_uncert * tmp_model_uncert + tmp_data_uncert * tmp_data_uncert)

        res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov

        norm *= 1 / np.sqrt(cov)

    res *= -0.5

    return norm * np.exp(res)


#
def posterior():
    return prior() * likelihood()


### Plot the Posterior ###
# Info about parameters
param1_label = paramNames[0]
param1_truth = paramTruths[0]

param2_label = paramNames[1]
param2_truth = paramTruths[1]

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(param1_label)
plt.ylabel(param2_label)

# Compute the posterior for a range of values of the parameter "x"
param1_range = np.arange(paramMins[0], paramMaxs[0], (paramMaxs[0] - paramMins[0]) / 100.)
param2_range = np.arange(paramMins[1], paramMaxs[1], (paramMaxs[1] - paramMins[1]) / 100.)

param1_mesh, param2_mesh = np.meshgrid(param1_range, param2_range, sparse=False, indexing='ij')

posterior_array = np.array([posterior()])

# Plot the posterior
cs = plt.contourf(param1_mesh, param2_mesh, posterior_array, 20)

cbar = plt.colorbar(cs, label="Posterior")

plt.plot([param1_truth], [param2_truth], "D", color='red', ms=10)

plt.tight_layout()
plt.show()

###############################
# Plotting marginal posterior #
###############################

# Posterior vs param_1
plt.figure()
plt.xscale('linear')
plt.yscale('linear')
# plt.xlim(0,2)
# plt.ylim(1e-5,1e2)
plt.xlabel(param1_label)
plt.ylabel(r'Posterior')

# The marginal posterior for a parameter is obtained by integrating over a subset of other model parameters

# Compute the posterior for a range of values of the parameter "param_1"
param1_range = np.linspace(paramMins[0], paramMaxs[0], 30)

posterior_list = np.array([scipy.integrate.quad(posterior(), paramMins[1], paramMaxs[1])[0]
                           for param1_val in param1_range])

plt.plot(param1_range, posterior_list, "-", color='black', lw=4)

plt.axvline(x=param1_truth, color='red')

plt.tight_layout()
plt.show()

# Posterior vs param_2
plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(param2_label)
plt.ylabel(r'Posterior')

# The marginal posterior for a parameter is obtained by integrating over a subset of other model parameters

# Compute the posterior for a range of values of the parameter "param_2"
param1_range = np.linspace(paramMins[1], paramMaxs[1], 30)

posterior_list = np.array([scipy.integrate.quad(posterior(), paramMins[0], paramMaxs[0])[0]
                           for param2_val in param2_range])

plt.plot(param2_range, posterior_list, "-", color='black', lw=4)

plt.axvline(x=param2_truth, color='red')

plt.tight_layout()
plt.show()
