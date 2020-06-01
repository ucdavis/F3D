""" 
This file contains the definition of all the functions that are 
involved in running the mcmc chain and plotting the resulting
trace and histogram plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from prior_likelihood import log_prior, log_likelihood


def metropolis(
    params, candidate_params, data, prior_func, likelihood_func, prior_mode="uniform"
):
    """
    this is the function that performs the metropolis-hastings algorithm.
    This function decides if we keep trials or not, it is called inside 
    of the mcmc chain algorithm. In this function, we compare the log 
    likehood value instead of the likelihood values -- so products/quotients
    of likelihoods become sums/differences.

    parameters
    ----------
    params : list of dimensions expected by likelihood_func/prior_func
        parameter values we are currently at

    candidate_params : list of dimensions expected by likelihood_func/prior_func
        potential new parameters values to move to

    data : whatever format data that likelihood_function needs as an 
        input the dataset we are running the MCMC on

    prior_func : function with inputs (params, magnitude_mode=arg)
        function that calculates the log of the prior probability of our
        set of parameters this function should return the string 'forbidden'
        if forbidden parameter ranges are entered.

    likelihood_func : function with inputs: (params, data)
        function that calculates the log likelihood of our set of parameters 
        given the data

    prior_mode : an input that is recognized by the prior function as 
        mag_mode=prior_mode. It tells you what prior to use for M 
        (uniform or gaussian).

    returns
    -------
    True : if we should accept the move to candidate params
    False : if we should reject the move to candidate params

    """

    # Checking if the candidate parameters are in the forbidden regime

    if prior_func(candidate_params, magnitude_mode=prior_mode) == "forbidden":
        return False

    else:

        # Function that calculate the posterior probability for the given set
        # of parameters.

        def get_log_prob(params):

            return prior_func(params, magnitude_mode=prior_mode) + likelihood_func(
                params, data
            )

        threshhold = np.exp(
            min(0, get_log_prob(candidate_params) - get_log_prob(params))
        )

        decide = np.random.uniform(0, 1, 1)

        if threshhold > decide:
            return True
        else:
            return False


def chain(
    data,
    max_trials=10000,
    convergence_window=50,
    convergence_threshhold=0.001,
    start_state=np.ones(4) + 1,
    gen_variances=np.ones(4) / 5,
    prior_func=log_prior,
    likelihood_func=log_likelihood,
    prior_mode="uniform",
):
    """
    this is the core function that makes our MCMC chain, it relies on the
    metropolis and convergence_test functions defined in this document.

    parameters
    ----------

    data :  whatever format data the likelihood function needs as an input
        the dataset we are running the MCMC on

    max_trials : int
        prevents it from taking too long if it gets stuck without convergence

    convergence_window : int
        how large a range it averages over for convergence

    convergence_threshhold : number>0 and < 1
        the maximum allowed percent change for reaching convergence, .01 means 1%

    start_state : list of dimensions expected by likelihood/prior functions
        initial values of all the cosmological parameters

    gen_variances : None, 1-D list or array, or 2D numpy array
        sets the variance for generating new samples using np.random.multivariate_normal
        if None: uses a hardcoded non-diagonal covariance matrix that was found
        empirically for case that only includes the statistical error
        if "systematic": uses a hardcoded non-diagonal covariance matrix that
        was found empirically for case that includes both the statistical and 
        systematic error with a gaussian prior over M.
        if "systematic_fix_M": uses a hardcoded non-diagonal covariance matrix that
        was found empirically for the case that includes both the statistical and 
        systematic error keeping M fixed.
        if 1-D list or array : uses a diagonal covariance matrix with diagonal
        elements = list elements
        if 2-D array : uses the 2D array as the covariance matrix

    prior_func : function with inputs (params, magnitude_mode=arg)
        function that calculates the prior probability of our set of parameters

    likelihood_func : function with inputs: (params, data)
        function that calculates the likelyhood of our set of parameters given the data

    prior_mode : an input that is recognized by the prior function as mag_mode=priorm_mode

    returns
    -------
    chn : numpy array of dimension [N, number of parameters]
        this is your MCMC chain, N is  2*convergence window< N< mat_trials
    rej : numpy array of dimension [N, number of parameters]
        these are the samples that got rejected by the algorithm,
        will have np.nan for the whole row if the trial was accepted
    convergence_value: numpy array of dimension start_state
        Return the values of the parameters at which the convergence
        happened. Return an empty array if convergence failed.
    """
    chain = []
    rejects = []
    current = start_state
    i = 0
    convergence = False

    # Calculating the covariance matrix. If gen_variances is provides as a
    # 1-d or 2-d array then use that for the covariance matrix. Otherwise
    # There are two hardcoded covariance matrices for the generating function,
    # for the cases of sys+stat error and one for just stat that are estimated
    # by looking at the covariance matrices of long chains generated by diagonal
    # covariance matrices.

    if gen_variances is None:
        covariance = 0.1 * np.array(
            [
                [0.015, 0.024, 0.070, 0.0],
                [0.024, 0.048, 0.177, 0.0],
                [0.070, 0.177, 1, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

    elif gen_variances == "systematic":
        covariance = np.array(
            [
                [2.282e-3, 2.729e-3, -3.856e-4, -7.165e-5],
                [2.729e-3, 4.202e-3, 1.005e-2, 1.713e-5],
                [-3.856e-4, 1.005e-2, 1.000e0, 2.662e-2],
                [-7.165e-5, 1.713e-5, 2.662e-2, 7.561e-4],
            ]
        )

    elif gen_variances == "systematic_fix_M":
        covariance = np.array(
            [
                [2.282e-3, 2.729e-3, -3.856e-4, 0],
                [2.729e-3, 4.202e-3, 1.005e-2, 0],
                [-3.856e-4, 1.005e-2, 1.000e0, 0],
                [0, 0, 0, 0],
            ]
        )

    elif len(np.shape(gen_variances)) == 1:
        covariance = np.diag(gen_variances)

    else:
        covariance = gen_variances

    # Start running the chain and end if you reach the maximum number
    # of trials or the chain converges.

    while convergence is False and i < max_trials:

        # generate the candidate parameters

        candidate = np.random.multivariate_normal(current, covariance)

        i += 1

        # Accept ot reject the candidate parameters according to the
        # metropolis-hastings algorithm

        if metropolis(
            current, candidate, data, prior_func, likelihood_func, prior_mode=prior_mode
        ):
            rejects.append(np.zeros(len(start_state)) * np.nan)
            current = candidate
        else:
            rejects.append(candidate)
        chain.append(current)

        # check if the chain coverged else keep running

        convergence, diff_booleans, convergence_value = convergence_test(
            chain, convergence_window, convergence_threshhold
        )

        # printing the progress

        print("done {:2.1%} of max trials".format(i / max_trials), end="\r")

    rej = np.asarray(rejects)
    chn = np.asarray(chain)

    # If the chain did not converge in the maximum number of trial, then say
    # the convergence failed and show which parameters haven't converged.

    if convergence is False:
        print("convergence failed. converged parameters:", diff_booleans)
        return chn, rej, None

    # if the chain has converged, then say the chain converged to whatever
    # parameter set the convergence occured at.

    else:
        print(
            "The chain has converged to the values:",
            convergence_value,
            "in {} trials".format(i),
        )
        return chn, rej, convergence_value


def convergence_test(chain, convergence_window, convergence_threshhold):
    """
    this function exists solely to be called inside of the chain function,
    and it does a simple convergence test where we compare the average of 
    the parameters over two non-overlapping windows of our most recent chain 
    data and claim convergence when those avergaes are equal to each other
    within a tolerance: convergence_threshold

    parameters
    ----------

    chain : a list of parameter values, dimension (some int, number of params )
        the mcmc chain we are testing

    convergence_window : int
        how large a range it averages over for convergence

    convergence_threshhold : number>0 and < 1
        the maximum allowed percent change for reaching convergence, .01 means 1%

    returns
    -------

    True or False : boolean
        True if the mean of the most recent L (L is given by the convergence_window)
        samples is within the threshold % of the mean over the previous L samples for
        all params. False if not any of the avergaes has changed by more than the 
        threshold %.

    diff_booleans : 1-D list of True/False, length equal to number of params
        True/False depending if the corresponding parameter has converged or not.

    new_means or []: list of dimension of the parameter set
        return the mean mean of the most recent L samples if convergence occured
        else return an empty array.
    """

    # Do convergence testing if the sample size is large enough for the
    # convergence testing.

    if len(chain) > 2 * convergence_window:

        # mean of the old L samples

        old_means = np.mean(
            chain[-2 * convergence_window + 1 : -convergence_window], axis=0
        )

        # mean of the most recent L samples

        new_means = np.mean(chain[-convergence_window:-1], axis=0)

        # Check for convergence

        diff_booleans = (
            abs(new_means - old_means) / abs(old_means) < convergence_threshhold
        )

        if sum(diff_booleans) == len(diff_booleans):
            return True, diff_booleans, new_means

        else:
            return False, diff_booleans, []

    # if the sample size is not large enough for the convergence testing,
    # simply return false.

    else:
        return False, [], []


def plot_chain_behaviour(
    chain,
    rejects,
    plot_rejects=True,
    one_d_hist_1=0,
    one_d_hist_2=1,
    two_d_hist_1=0,
    two_d_hist_2=1,
    one_d_bins=30,
    two_d_bins=100,
    two_d_histogram=True,
    save=False,
):
    """
    this function is for plotting trace plots of all 4 parameters, and
    1-D/2D histograms of w/e 2 paramters we want.

    parameters
    ----------
    chain : numpy array
        the chain we are plotting
    rejects : numpy array, same shape as chain
        the rejected samples from the chain
    plot_refects = True/False
        False if you dont want to plot rejects
    one_d_his_1, one_d_his_2, two_d_his1, two_d_his_2 : ints
        these are the indices of the parameters you want to plot in the
        histograms (default is 0,1 for the two omegas)
    one_d_bins, two_d_bins : ints
        number of bins for our 1d and 2d histograms
    two_d_histogram : True/False
        if False, we plot a scatterplot instead of histogram
    save : True/False
        True for saving the plot

    returns
    -------
    shows and/or saves plots, no returns
    """

    od1 = one_d_hist_1
    od2 = one_d_hist_2
    td1 = two_d_hist_1
    td2 = two_d_hist_2

    names = dict(
        [(0, "$\\Omega_m$"), (1, "$\\Omega_\\Lambda$"), (2, "$H_0$"), (3, "$M$")]
    )

    plt.rc("axes", titlesize=18)
    plt.rc("axes", labelsize=18)
    plt.rc("figure", titlesize=20)

    fig, ax = plt.subplots(3, 2, figsize=(20, 15))

    hist_or_scatter = dict([(True, "histogram"), (False, "scatter plot")])

    fig.suptitle(
        "plots 1-4 are trace plots, 5 is a 1D historgram of 1 or 2 parameters and 6 is a 2D "
        + hist_or_scatter[two_d_histogram]
    )

    ax[0, 0].plot(chain[:, 0])
    ax[0, 0].set_title(names[0])
    ax[0, 1].plot(chain[:, 1])
    ax[0, 1].set_title(names[1])
    ax[1, 0].plot(chain[:, 2])
    ax[1, 0].set_title(names[2])
    ax[1, 1].plot(chain[:, 3])
    ax[1, 1].set_title(names[3])

    if plot_rejects:
        rej_alpha = 400 / len(rejects[:, 0])
        ax[0, 0].plot(rejects[:, 0], "+", alpha=rej_alpha)
        ax[0, 1].plot(rejects[:, 1], "+", alpha=rej_alpha)
        ax[1, 0].plot(rejects[:, 2], "+", alpha=rej_alpha)
        ax[1, 1].plot(rejects[:, 3], "+", alpha=rej_alpha)

    # when doing the averages, we drop the first 25% of samples, cchn is
    # the chain with the first 25% of samples removed.

    cutoff = int(len(chain[:, 0]) / 4)
    cchn = chain[cutoff:, :]

    mu1 = np.mean(cchn[:, od1])
    mu2 = np.mean(cchn[:, od2])
    std1 = np.std(cchn[:, od1])
    std2 = np.std(cchn[:, od2])

    mean_names = dict(
        [
            (0, "$\\overline{\\Omega}_m$"),
            (1, "$\\overline{\\Omega}_\\Lambda$"),
            (2, "$\\overline{H}_0$"),
            (3, "$\\overline{M}$"),
        ]
    )

    ax[2, 0].hist(cchn[:, od1], bins=one_d_bins, density=1)
    ax[2, 0].axvline(mu1, color="k")

    ax[2, 0].text(mu1, 0, mean_names[od1] + "={:.3f}".format(mu1), va="bottom")
    ax[2, 0].set_title(mean_names[od1] + "$={:.3f}\\pm{:.3f}$".format(mu1, std1))

    if od2 is not None:

        ax[2, 0].hist(cchn[:, 1], bins=one_d_bins, density=1)
        ax[2, 0].axvline(mu2, color="k")
        ax[2, 0].text(np.mean(cchn[:, od2]), 0, mean_names[od2] + "={:.3f}".format(mu2))
        ax[2, 0].set_title(
            mean_names[od1]
            + " $={:.3f}\\pm{:.3f}$ ".format(mu1, std1)
            + mean_names[od2]
            + " $={:.3f}\\pm{:.3f}$ ".format(mu2, std2)
        )

    if two_d_histogram:
        p_range = np.array(
            [
                [min(cchn[:, td1]), max(cchn[:, td1])],
                [min(cchn[:, td2]), max(cchn[:, td2])],
            ]
        )
        ex_range = np.zeros((2, 2))
        L = 0.2 * (p_range[:, 1] - p_range[:, 0])
        ex_range[:, 0], ex_range[:, 1] = p_range[:, 0] - L, p_range[:, 1] + L
        ax[2, 1].hist2d(
            cchn[:, td1],
            cchn[:, td2],
            bins=two_d_bins,
            range=[[ex_range[0, 0], ex_range[0, 1]], [ex_range[1, 0], ex_range[1, 1]]],
            cmap="BuGn",
        )
    else:
        ax[2, 1].scatter(cchn[:, td1], cchn[:, td2], alpha=0.05)

    ax[2, 1].set_xlabel(names[td1])
    ax[2, 1].set_ylabel(names[td2])

    if save:
        plt.savefig("chain{}.png".format(len(chain[:, 0])))

    plt.show()


def estimate_covariance(chain, scaling=1, trim_ratio=0.25):
    """
    This function calculates the hardcoded covariance matrix used
    for the covariance matrix of the generating function.

    params
    -----
    chain: an array where the variables are different columns and
    rows are observations
        we will estimate the covariance b/w these variables for this
        data set
    scaling: float
        scale the maximum value in the covariance matrix to be this
        number, generally <=1
    trim_ratio: float > 0 and < 1
        this is the ratio of data that we want to drop before looking
        at covariance

    returns
    -------
    cov: N X N np array, N is the number of columns in the input chain
        this is the covariance matrix
    """

    # Drop the initial few data points according to the trim ratio

    cutoff = int(trim_ratio * len(chain[:, 0]))

    # Calculate the covariance matrix using the chain

    cchn = chain[cutoff:, :]
    cov = np.cov(cchn, y=None, rowvar=False)

    return scaling * cov / np.max(np.abs(cov))
