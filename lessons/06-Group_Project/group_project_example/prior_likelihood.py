""" Defines the prior and likelihood functions.

This file contains the functions for the log prior and log 
likelihood functions needed to calculate the posterior
distribution over the cosmological parameters.
"""

import numpy as np
import pandas as pd
import math

# Here we import the apparent_mag function.

from theoretical_mag import calculate_apparent_mag


def log_prior(params, magnitude_mode="uniform"):

    """ This function calculates the log prior.

    Parameter:
    1. params - This is a list of the 4 cosmological parameters
    omega_m, omega_lambda, H_0 and M.
    2. magnitude_mode - This parameter decides whether the prior
    on M (absolute magnitude) is 'uniform' or 'M_gaussian'.

    omega_m, omega_lambda and H_0 cannot take negative values therefore
    we choose a regime to be forbidden when at least one of 
    the three parameters goes negative( or when omega_m and/or 
    omega_lambda gets higher than 2.5). We also don't allow any values
    of omega_m and omega_lambda that will cause the expression

    Omega_M(1+z)^3 + Omega_Lambda + Omega_K(1+z)^2

    to go negative because we have to calculate the sqaure root of the
    above expression in the apparent magnitue function. For any other 
    values of omega_m, omega_lambda, and H_0, we choose a uniform prior 
    probabiility.
    """

    # Forbidden regime if omega_m and/or omega_lambda and/or H_0 are negative.

    if any(i < 0 for i in params[0:-1]):
        return "forbidden"

    # Forbidden regime if omega_m and/or omega_lambda are greater than 2.5.

    if any(i > 2.5 for i in params[0:2]):
        return "forbidden"

    # Forbidden regime if Omega_M(1+z)^3 + Omega_Lambda + Omega_K(1+z)^2 goes
    # negative for the maximum redshift value in the Supernovae data.

    z_max = 1.62

    if (
        params[0] * (1 + z_max) ** 3
        + params[1]
        + (1 - params[0] - params[1]) * (1 + z_max) ** 2
        < 0
    ):
        return "forbidden"

    # Uniform prior

    if magnitude_mode == "uniform":
        return 0

    # Gaussian prior on corrected supernova absolute magnitude of
    # M =19.23 +/- 0.042.

    else:
        return -0.5 * pow((params[3] + 19.23) / 0.042, 2)


def log_likelihood(params, data_lcparam, sys_error=None):

    """This function calculates the log likelihood.

    Parameter:
    1. params - This is a list of the 4 cosmological parameters
    omega_m, omega_lambda, H_0 and M.
    2. data_lcparam - Importing the data file that contains the
    redshift, aparent magnitude and the statistical error data.
    3. sys_error - This is a 40x40 matrix that contains the 
    systematic error data. The default value for this argument is
    None. This means that if this argument isn't passed into this
    function then the systematic error isn't included in the
    covariance matrix calculation.
    """

    # Importing an array of size 40 that contains the apparent
    # magnitude data.

    app_mag = pd.Series.to_numpy(data_lcparam.mb)

    # Calculating the difference between the measured (app_mag)
    # and estimated apparent magnitude (calculate_apparent_mag).

    diff_app_mag = app_mag - calculate_apparent_mag(params, data_lcparam.zcmb)

    # Defining a 40x40 diagonal matrix whose diagonal entries
    # are the square of the corresponding statistical error.

    stat_error = np.diag(pow(pd.Series.to_numpy(data_lcparam.dmb), 2))

    # Only include the statistical error in the covariance
    # matrix calculation

    if sys_error is None:
        inv_cov_matrix = np.linalg.inv(stat_error)

    # Include the systematic error as well in the covariance
    # matrix calculation.

    else:
        inv_cov_matrix = np.linalg.inv(stat_error + sys_error)

    # return the calculate log likelihood value.

    return -0.5 * (diff_app_mag @ inv_cov_matrix @ diff_app_mag)
