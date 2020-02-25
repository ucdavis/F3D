import sys
import numpy as np


def main(significant_digits, n_darts_per_scoring):
    """
    Calculate pi and print to STDOUT.

    Parameters
    ----------
    significant_digits: int
        The number of significant digits to calculate pi out to.
    n_darts_per_scoring: int
        The number of darts to throw between scores. Higher numbers
        converge faster.
    """
    pi = calculate_pi(significant_digits, n_darts_per_scoring)
    print(pi)


def calculate_pi(
        significant_digits,
        num_darts_thrown=0,
        num_darts_in_circle=0,
        history=None,
        n_darts_per_scoring=100):
    """
    Calculates pi.

    Parameters
    ----------
    significant_digits: int
        The number of significant digits to calculate pi out to.
    num_darts_thrown: int (default 0)
        The number of darts thrown.
    num_darts_in_circle: int (default 0)
        The number of darts that have landed in the circle.
    history: list(float) or NoneType
        Any past estimates of `pi`, used to check convergence. If None,
        Assumed to be the empty list.
    n_darts_per_scoring: int
        The number of darts to throw between scores. Higher numbers
        converge faster.
    Returns
    -------
    pi: float
        The value of pi to `significant_digits`.
    """
    history = history or list()
    threshold = get_threshold(significant_digits)
    # We set the recursion limit because this function calls itself many times
    # depending on threshold.
    sys.setrecursionlimit(10**(significant_digits + 2))
    num_darts_thrown, num_darts_in_circle = _throw_and_score(
        n_darts_per_scoring, num_darts_thrown, num_darts_in_circle)
    result = 4 * num_darts_in_circle / num_darts_thrown
    history.append(result)
    if not is_converged(threshold, history):
        calculate_pi(significant_digits, num_darts_thrown, num_darts_in_circle, history)
    return round(history[-1], significant_digits - 1)


def _throw_and_score(n_darts_per_scoring, num_darts_thrown, num_darts_in_circle):
    """
    Throw `n_darts_per_scoring` and increment the running count of
    total darts and ones inside the circle.

    n_darts_per_scoring: int
        The number of darts to throw between scores. Higher numbers
        converge faster.
    num_darts_thrown: int (default 0)
        The number of darts thrown.
    num_darts_in_circle: int (default 0)
        The number of darts that have landed in the circle.

    Returns
    -------
    num_darts_thrown: int
        The incremented total number of darts thrown
    num_darts_in_circle: int
        The incremented total number of darts in the circle
    """
    num_darts_thrown += n_darts_per_scoring
    num_darts_in_circle += get_number_in_circle(n_darts_per_scoring)
    return num_darts_thrown, num_darts_in_circle


def get_threshold(significant_digits):
    """
    Get the maximum threshold two estimates should differ by if we've converged
    up to `significant digits`.

    Parameters
    ----------
    significant_digits: int
        The significant digits we want to calculate pi to.

    Returns
    -------
    threshold: float
        The corresponding threshold.
    """
    return 1 / (10 ** (significant_digits + 1))


def is_converged(threshold, history):
    """
    Return whether the threshold has converged.

    Parameters
    ----------
    threshold: float
        The threshold to test against.
    history:
        The last 100 historical estimates. We say we have converged when
        the difference between all consecutive histories is less than ``threshold``.

    Returns
    -------
    converged: bool
        Whether we have converged.
    """
    if len(history) < 100:
        return False
    return np.all(abs(np.diff(history[-100:])) < threshold)


def get_number_in_circle(num_thrown):
    """
    Given ``n_thrown`` darts randomly thrown at the outer tangent circle, return
    how many are inside the unit circle.

    Parameters
    ----------
    num_thrown: int
        the number of darts thrown.

    Returns
    -------
    num_in_circle: float
        The number of darts the hit in the circle.
    """
    coords = get_random_coords(num_thrown)
    return int(sum([np.linalg.norm(coord) < 1 for coord in coords]))


def get_random_coords(num_thrown):
    """
    Get landing coordinates for `num_thrown` darts.

    Parameters
    ----------
    num_thrown: int
        the number of darts thrown.

    Returns
    -------
    coords: np.ndarray
        The coordinates of the darts.
    """
    return np.random.random(2 * num_thrown).reshape(num_thrown, 2)
