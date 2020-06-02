import numpy as np 
from matplotlib import pyplot as plt

class CalculateE:
    """
    This is a class for finding the best value of e given
    data sampled from a random normal distribution

    Example use:

    Calculate the best value of a using metropolis hastings
        calculator = calculate_e.CalculateE(nsamples=200)
        calculator.run_mcmc(1.01, 10) 
        calculator.plot(mcmc=True)
    Calculate the best value of a using a uniform sampling of parameter space
        calculator = calculate_e.CalculateE(nsamples=200)
        calculator.uniformly_sample_space(1.01, 7)  
        calculator.plot(mcmc=False)
    """
    def __init__(self, nsamples=100):
        """
        initialize calculate_e object with nsamples of data

        Inputs:
        ---------
        nsamples: positive int
            number of data samples
        """
        self.nsamples = nsamples
        self.ntrials = 1000
        self.data = self._generate_data(nsamples)
        self.a_array = None
        self.posterior = None
        self.chain = None
        

    def _generate_data(self, nsamples):
        '''
        Generate samples according to a gaussian distribution with mean 0 and stddev 1
        Inputs:
        --------
            nsamples:int
                number of samples
        Returns:
        --------
            list of samples of length nsamples
        '''
        return np.random.standard_normal(nsamples)

            
    def calculate_posterior(self, a):
        '''
        Calculate the posterior by calculating P(a|x) for each x
        Inputs:
        --------
            a: float
                independent variable
        '''
        Na = np.sqrt(np.log(a)/(2*np.pi))
        return np.product(Na * a**(-(self.data**2)/2))

    def plot(self, mcmc=True):
        """
        Inputs:
        --------
        mcmc: bool
            defines which type of plot should be given
        Produce validation plots
        If mcmc is False then a is plotted against the posterior
        If mcmc is True then the chain is plotted against trial number
        """
        fig, ax = plt.subplots(1, 1)
        if mcmc is False:
            ax.plot(self.a_array, self.posterior, label='Posterior')
            ax.axvline(np.e, color='C1', label='True value')
            ax.set_xlabel('Parameter a')
            ax.legend()
            ax.set_ylabel('P(a|x)')
            ax.set_title('Probability of a given a uniformly sampled parameter space')
        if mcmc is True:
            ax.plot(np.arange(len(self.chain))+1, self.chain, label='chain')
            ax.axhline(np.e, color='C1', label='True value')
            ax.legend()
            ax.set_xlabel('trial number')
            ax.set_ylabel('a')
            ax.set_title('Values of a using Metropolis Hastings algorithm')

    def uniformly_sample_space(self, a_min, a_max, ntrials=1000):
        """
        uniformaly sample the parameter space from a_min to a_max with ntrials points
        and find the posterior at each point

        Inputs:
        ---------
        a_min: float
            minimum value of a (inclusive); must be greater than 1
        a_max: float
            maximum value of a (exclusive)
        ntrials: int
            number of points used between a_min and a_max

        Sets:
        self.a_array: array
            array of parameter values uses
        self.posterior: array
            array of posterior values corresponding to a_array values
        """
        self._check_inputs(a_min, a_max, ntrials)
        if ntrials:
            self.ntrials = ntrials
        else:
            ntrials = self.ntrials
        self.a_array = np.linspace(a_min, a_max, ntrials)
        self.posterior = np.array([self.calculate_posterior(a) for a in self.a_array])

    def run_mcmc(self, a_min, a_max, ntrials=1000):
        """
        Run the metropolis hastings routine on the posterior to calculate a the
        best value of a
        Inputs:
        -------
        a_min: float
            minimum value of a (inclusive); must be greater than 1
        a_max: float
            maximum value of a (exclusive)
        ntrials: int
            number of points used between a_min and a_max

        Sets:
        self.chain : array-like
            values of a in metropolis hastings chain
        """
        self._check_inputs(a_min, a_max, ntrials)
        if ntrials:
            self.ntrials = ntrials
        else:
            ntrials = self.ntrials
        self.metropolis_hastings(ntrials, a_min, a_max)


    def _check_inputs(self, a_min, a_max, ntrials):
        """
        Checks the validity of values input for a_min, a_max, and ntrials
        """
        assert a_min > 1, 'due to normalization of posterior, a must be greater than 1'
        assert a_min < a_max
        assert (ntrials > 0) and isinstance(ntrials, int), 'ntrials must be a positive integer'

    def restart_chain(self):
        """
        Initialize chain to start metropolis hastings with a new starting point
        """
        self.chain=None

    def metropolis_hastings(self, ntrials, a_min, a_max):
        """
        Assumes generate function is uniform distributed from 1 to 10
        """
        a0 = np.random.uniform(a_min, a_max, size=1)
        prob_a0 = self.calculate_posterior(a0)
        trial_num = 1
        a_current = a0
        if self.chain is None:
            self.chain = [a_current,]
        prob_a_current = prob_a0
        while trial_num <= ntrials:
            a_trial = np.random.uniform(1, 10, size=1)
            prob_a_trial = self.calculate_posterior(a_trial)
            acceptance_prob = min(1, prob_a_trial/prob_a_current)
            #Accept or reject
            u = np.random.uniform(0,1)
            if u <= acceptance_prob:
                a_current = a_trial
                prob_a_current = prob_a_trial
            trial_num +=1
            self.chain.append(a_current[0])
