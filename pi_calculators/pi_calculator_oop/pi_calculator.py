import numpy as np
from .dart import Dart
from .utils import _get_random_pairs


class PiCalculator:
    """
    Calculate pi to ``significant_digits`` using the
    dart method.

    Pi is the ratio of darts thrown at random in a
    square of side length 2 that fall inside a unit
    circle centered at the origin.

    Attributes
    ----------
    darts_per_score: int
        The number of darts to throw between scoring.
    history: list(float)
        The history of the iterative process.
    converged: bool
        True if the last 100 trials differ by less than
        a threshold set by ``significant_digits``
    """

    def __init__(self, darts_per_score):
        self.darts_per_score = darts_per_score
        self._history = [0, 0]
        self.significant_digits = None  # set during calculate()

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history.append(value)

    @property
    def converged(self):
        """
        Check if the result has converged to ``significant_digits`` number
        of places for the past 100 trials.

        Parameters
        ----------
        significant_digits: int
            Number of significant digits to use to set the convergence threshold.
        """
        threshold = 1 / 10**(self.significant_digits + 1)
        if len(self._history) < 100:
            return False
        return np.ptp(self._history[-100:]) < threshold

    def throw_and_score(self):
        """
        Throw ``darts`` at the unit circle, and count
        how many land inside the unit circle.

        Parameters
        ----------
        n_darts: int (default 100)
            The number of darts to throw

        Returns
        -------
        score: int
            The number of darts that landed inside
            the unit circle.
        """
        coords = _get_random_pairs(self.darts_per_score)
        darts = map(Dart, coords)
        return len([d for d in darts if d.in_circle])

    def calculate(self, significant_digits):
        """
        Calculate pi to `significant digits`.

        Parameters
        ----------
        significant_digits: int
            The number of significant digits in the result
        """
        self.significant_digits = significant_digits
        total_thrown = 0
        in_circle = 0
        while not self.converged:
            total_thrown += self.darts_per_score
            in_circle += self.throw_and_score()
            pi_approx = 4 * in_circle/total_thrown
            self.history = pi_approx

        return round(self.history[-1], significant_digits - 1)
