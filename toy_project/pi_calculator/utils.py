import numpy as np


def _get_random_pairs(num_pairs):
    """
    Get ``num_pairs`` tuples of pairs of random numbers
    between [-1, 1].

    Parameters
    ----------
    num_pairs: int
        The number of tuples

    Returns
    -------
    pairs: np.array
        The random pairs.
    """
    # generate 2 * num_tuples random numbers in a numpy
    # array
    rand_nums = 2 * (np.random.random(2 * num_pairs)) - 1
    return rand_nums.reshape(num_pairs, 2)
