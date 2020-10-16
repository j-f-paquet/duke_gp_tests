import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate

import warnings

warnings.filterwarnings('ignore')


def ch_mult(params):
    norm = params['norm']
    es_eff = params['eta_over_s_eff']
    result = 100 * norm * (1 + es_eff)
    return result


def ch_v2(params):
    norm = params['norm']
    es_eff = params['eta_over_s_eff']
    result = 0.04 / (1 + 6 * es_eff) + 0.0003 * norm
    return result


param_d = {
    'norm': {
        "label": "Normalization of energy deposition",
        "range": [6, 12],
        "truth": 10.
    },
    'eta_over_s_eff': {
        "label": r"Effective $\eta/s$",
        "range": [.01, .3],
        "truth": 0.12
    }
}
obs_d = {
    "multiplicity": {
        'fct': ch_mult,
        'label': r'$dN_{ch}/d\eta$',
        'fake_exp_rel_uncert': 0.05,
        'theoretical_relative_uncertainty': 0.05
    },
    "v2": {
        'fct': ch_v2,
        'label': r'$v_2\{2\}$',
        'fake_exp_rel_uncert': 0.05,
        'theoretical_relative_uncertainty': 0.05
    },
}

data_d = {}
for obs_name, info_d in obs_d.items():
    obs_fct = info_d['fct']
    data_d[obs_name] = {}
    param_value_list = {item: tmp_d['truth'] for (item, tmp_d) in param_d.items()}
    tmp_value = obs_fct(param_value_list)
    data_d[obs_name]['mean'] = tmp_value
    fake_exp_rel_uncert = info_d['fake_exp_rel_uncert']
    data_d[obs_name]['uncert'] = fake_exp_rel_uncert * tmp_value

for obs_name, info_d in obs_d.items():
    obs_fct = info_d['fct']
    obs_label = info_d['label']
    tmp_data_d = data_d[obs_name]
    data_mean = tmp_data_d['mean']
    data_uncert = tmp_data_d['uncert']
    param_name_list = list(param_d.keys())
    x_param_name = param_name_list[0]
    xmin, xmax = param_d[x_param_name]['range']
    x_label = param_d[x_param_name]['label']
    y_param_name = param_name_list[1]
    ymin, ymax = param_d[y_param_name]['range']
    y_label = param_d[y_param_name]['label']

    plt.figure()
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x_range = np.arange(xmin, xmax, (xmax - xmin) / 100.)
    y_range = np.arange(ymin, ymax, (ymax - ymin) / 100.)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range, sparse=False, indexing='ij')
    print(np.shape(x_mesh), np.shape(y_mesh))
    z_list = [obs_fct({x_param_name: x_val, y_param_name: y_val}) for (x_val, y_val) in zip(x_mesh, y_mesh)]
    print(np.shape(z_list))
    cs = plt.contourf(x_mesh, y_mesh, z_list, 20)
    cbar = plt.colorbar(cs, label=obs_label)
    plt.tight_layout()
    plt.show()

calc_d = {}
for obs_name, info_d in obs_d.items():
    nde = 20
    obs_fct = info_d['fct']
    param_name_list = list(param_d.keys())
    x_param_name = param_name_list[0]
    xmin, xmax = param_d[x_param_name]['range']
    y_param_name = param_name_list[1]
    ymin, ymax = param_d[y_param_name]['range']
    x_list = np.linspace(xmin, xmax, num=nde)
    y_list = np.linspace(ymin, ymax, num=nde)
    x_mesh, y_mesh = np.meshgrid(x_list, y_list, sparse=False, indexing='ij')
    calc_mean_list = [obs_fct({x_param_name: x_val, y_param_name: y_val}) for (x_val, y_val) in zip(x_mesh, y_mesh)]
    relative_uncertainty = info_d['theoretical_relative_uncertainty']
    calc_uncert_list = np.multiply(calc_mean_list, relative_uncertainty)
    calc_d[obs_name] = {'x_list': x_list, 'y_list': y_list, 'mean': calc_mean_list, 'uncert': calc_uncert_list}

emul_d = {}
for obs_name, info_d in obs_d.items():
    emul_d[obs_name] = {
        'mean': scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'],
                                           np.transpose(calc_d[obs_name]['mean']), kind='linear', copy=True,
                                           bounds_error=False, fill_value=None),
        'uncert': scipy.interpolate.interp2d(calc_d[obs_name]['x_list'], calc_d[obs_name]['y_list'],
                                             np.transpose(calc_d[obs_name]['uncert']), kind='linear', copy=True,
                                             bounds_error=False, fill_value=None)
    }


def prior():
    return 1


def likelihood(params, data):
    res = 0.0
    norm = 1.0
    for obs_name2 in obs_d.keys():
        emul_calc = emul_d[obs_name2]['mean']
        emul_uncert = emul_d[obs_name2]['uncert']
        data_mean2 = data[obs_name2]['mean']
        data_uncert2 = data[obs_name2]['uncert']
        emul_calc_vec = np.vectorize(emul_calc)
        emul_uncert_vec = np.vectorize(emul_uncert)
        param_name_list2 = list(param_d.keys())
        x_param_name2 = param_name_list2[0]
        x_value = params[x_param_name2]
        y_param_name2 = param_name_list2[1]
        y_value = params[y_param_name2]

        tmp_model_mean = emul_calc_vec(x_value, y_value)
        tmp_model_uncert = emul_uncert_vec(x_value, y_value)
        tmp_data_mean = data_mean2
        tmp_data_uncert = data_uncert2
        cov = (tmp_model_uncert * tmp_model_uncert + tmp_data_uncert * tmp_data_uncert)
        res += np.power(tmp_model_mean - tmp_data_mean, 2) / cov
        norm *= 1 / np.sqrt(cov)
    res *= -0.5
    return norm * np.exp(res)


def posterior(params, data):
    return prior() * likelihood(params, data)


param_name_list = list(param_d.keys())
x_param_name = param_name_list[0]
xmin, xmax = param_d[x_param_name]['range']
x_truth = param_d[x_param_name]['truth']
x_label = param_d[x_param_name]['label']
y_param_name = param_name_list[1]
ymin, ymax = param_d[y_param_name]['range']
y_truth = param_d[y_param_name]['truth']
y_label = param_d[y_param_name]['label']

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(x_label)
plt.ylabel(y_label)

x_range = np.arange(xmin, xmax, (xmax - xmin) / 100.)
y_range = np.arange(ymin, ymax, (ymax - ymin) / 100.)
x_mesh, y_mesh = np.meshgrid(x_range, y_range, sparse=False, indexing='ij')
posterior_array = [posterior({x_param_name: x_val, y_param_name: y_val}, data_d) for (x_val, y_val) in
                   zip(x_mesh, y_mesh)]
cs = plt.contourf(x_mesh, y_mesh, posterior_array, 20)
cbar = plt.colorbar(cs, label="Posterior")

plt.plot([x_truth], [y_truth], "D", color='red', ms=10)
plt.tight_layout()
plt.show()

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(x_label)
plt.ylabel(r'Posterior')

x_range = np.arange(xmin, xmax, (xmax - xmin) / 100.)
posterior_list = [
    scipy.integrate.quad(lambda y_val: posterior({x_param_name: x_val, y_param_name: y_val}, data_d), ymin, ymax)[0] for
    x_val in x_range]

plt.plot(x_range, posterior_list, "-", color='black', lw=4)
plt.axvline(x=x_truth, color='red')
plt.tight_layout()
plt.show()

plt.figure()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel(y_label)
plt.ylabel(r'Posterior')

y_range = np.arange(ymin, ymax, (ymax - ymin) / 100.)
posterior_list = [
    scipy.integrate.quad(lambda x_val: posterior({x_param_name: x_val, y_param_name: y_val}, data_d), xmin, xmax,
                         limit=100, epsrel=1e-4)[0] for y_val in y_range]

plt.plot(y_range, posterior_list, "-", color='black', lw=4)
plt.axvline(x=y_truth, color='red')
plt.tight_layout()
plt.show()
