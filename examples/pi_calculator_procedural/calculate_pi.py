'''
This script uses the idea that given a unit circle inside a unit square, the ratio of
the area of the circle to the area of the square is pi/4
'''

import numpy as np
import sys

### Parse Arguments ###
N_SIG_DIGS = int(sys.argv[1])
N_DARTS_PER_THROW = int(sys.argv[2])
if len(sys.argv) > 3:
    raise ValueError("Too many values passed!")

#    Given as input number of significant digits desired and number of darts per "throw", this program calculates pi.
#
#    It does so by randomly sampling from the unit square ("throwing darts"), and determining how many of those points
#    are in the unit circle that has the same center.
#
#    At least 100 throws are taken, and the pi estimate is updated each time. The most recent estimate uses the result of all
#    previous throws. More throws are taken for as long as needed to achieve convergence.
#
#    Convergence is achieved when fluctuations in the past 100 calculations of pi are sufficiently small, given the number of
#    desired significant figures.
#
#    ----------
#    N_SIG_DIGS: int
#        The number of significant digits to calculate pi out to.
#    N_DARTS_PER_THROW: int
#        The number of darts thrown at one time, for which an updated calculation is made.

### Setup ###
threshold = 1 / (10 ** N_SIG_DIGS + 1)

#initialize variables used in while loop
history = []
n_darts_thrown = 0
n_darts_in_circle = 0

converged = False

### Calculate Pi ###
while not converged:
    # Throw some darts and add them to the count
    n_darts_thrown += N_DARTS_PER_THROW
    # Create x,y coordinate pairs for new darts
    coords = np.random.rand(
        2 * N_DARTS_PER_THROW).reshape(N_DARTS_PER_THROW, 2)
    # Find the distance of each dart from the center of the circle
    distance = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
    # Add new darts that are inside unit circle to thrown to total
    n_darts_in_circle += np.size(distance[distance<1])
    # Recalculate guess based on N_DARTS_PER_THROW more darts
    pi_guess = 4 * n_darts_in_circle / n_darts_thrown
    # Include the resulting value of `pi` in the history
    history.append(pi_guess)
    # let's say we're converged if the last 100 trials differ
    # by less than threshold.
    if len(history) < 100: # Don't check for convergence until there are 100 entries in history
        converged = False
    else:
        converged = np.ptp(history[-100:]) < threshold

### Print Result to STDOUT ###
print(round(history[-1], N_SIG_DIGS - 1))
