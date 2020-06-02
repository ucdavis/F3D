#Bayesian e_Calculator

This exercise in Bayesian inference is to implement another (admittedly not very practical) method for calculating, or rather inferring the posterior probability density of e (the base of the natural logarithm) given some data. We then proceed, in a rather gratuitous application of the Metropolis-Hasting algorithm, to sample from this posterior density to form a Markov chain. We then use the Markov chain to estimate the posterior density. I call this application 'rather gratuitous' because we already know the posterior density and can plot it and do anything we want with it. The utility of generating a Markov chain will become more evident in the group project. 

1) Assume a Gaussian random number generator with zero mean and unit variance is producing samples, x,  with posterior density P(x|a) = N(a) a^{-x^2/2}. Analytically find N(a) so that P(x|a) is appropriately normalized.

3) Use Bayes’s theorem to calculate P(a|{x}) when one has multiple samples drawn; i.e. {x} =  (x_1, x_2, x_3, ... x_n).

4) Draw 100 samples and plot the resulting P(a|{x}).

5) Use the Metropolis Hastings algorithm to sample from this posterior and create a Markov chain.

6) Plot a "trace plot" which is sample number vs. parameter value.

7) Plot a histogram and compare with P(a|x).

Develop in the VS Code IDE under version control on your own GitHub repo. Submit a link to the GitHub repo.

 

#UPDATE: 

1) In (3), simply adopt a uniform prior, which means P(a) is independent of a.

2) In (3), you only need to calculate an un-normalized P(a|x); i.e., don't concern yourself with factors that have no dependence on a. If you do want to normalize P(a|x) you can do so by making sure that \int da P(a|x)=1 -- but you don't have to.

 

#Note for next year
Change this to have them estimate the variance of the distribution from which the dataset was drawn, rather than e. Change "plot a histogram" to "plot a histogram of the chain with variance as the x axis". Add "indicate in your graph the true value of the variance."