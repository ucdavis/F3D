'''
This script uses the idea that given a unit circle inside a unit square, the ratio of
the area of the circle to the area of the square is pi/4
'''

import numpy as np
import sys

### Parse Arguments ###
N_SIG_DIGS = sys.argv[1]
N_DARTS_PER_THROW = sys.argv[2]
#Raise exception if too many arguments are passed in
if len(sys.argv) > 2:
    raise ValueError("Too many values passed!")


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
    coords = np.random.random(2 * N_DARTS_PER_THROW).reshape(N_DARTS_PER_THROW, 2)
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
        converged = np.all(abs(np.diff(history[-100:])) < threshold)

### Print Result to STDOUT ###
print(round(history[-1], N_SIG_DIGS - 1))
