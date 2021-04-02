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
#          design points file name, [number of design points]

savedValues = np.load("paramsSmallTest.npy", allow_pickle=True)

