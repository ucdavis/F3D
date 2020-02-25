import numpy as np
import sys


### Parse Arguments ###
N_SIG_DIGS = sys.argv[1]
N_DARTS_PER_THROW = sys.argv[2]
if len(sys.argv) > 2:
    raise ValueError("Too many values passed!")


### Setup ###
threshold = 1 / (10 ** N_SIG_DIGS + 1)

history = list()
n_darts_thrown = 0
n_darts_in_circle = 0

converged = False

### Calculate Pi ###
while not converged:
    # Throw some darts and add them to the count
    n_darts_thrown += N_DARTS_PER_THROW
    coords = np.random(2 * N_DARTS_PER_THROW).reshape(N_DARTS_PER_THROW, 2)
    n_darts_in_circle += sum(np.linalg.norm(coords, axis=1) < 1)
    # Include the resulting value of `pi` in the history
    history.append(4 * n_darts_in_circle / n_darts_thrown)
    # let's say we're converged if the last 100 trials differ
    # by less than threshold.
    if len(history) < 100:
        converged = False
    else:
        converged = np.all(abs(np.diff(history[-100:])) < threshold)

### Print Result to STDOUT ###
print(round(history[-1], N_SIG_DIGS - 1))
