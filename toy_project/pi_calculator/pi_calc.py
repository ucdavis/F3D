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
    significant_digits: int
        The number of significant digits you need pi
        calculated to.

    history: list(float)
        The history of the iterative process.

    converged: bool
        True if the last 100 trials differ by less than
        a threshold set by ``significant_digits``
    """
    def __init__(self, significant_digits):
        self.significant_digits = significant_digits
        self._threshold = 1 / (10**(significant_digits+1))
        self._history = [0, 0]

    @property
    def history(self):
        return self._history


    @history.setter
    def history(self, value):
        self._history.append(value)

    @property
    def converged(self):
        if len(self._history) < 100:
            return False
        return np.all(abs(np.diff(self.history[-100:])) < self._threshold)

    @staticmethod
    def throw_and_score(n_darts=100):
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
        coords = _get_random_pairs(n_darts)
        darts = map(Dart, coords)
        return len([d for d in darts if d.in_circle])

    def __call__(self, darts_per_round=100):
        """
        Calculate pi.
        """
        total_thrown = 0
        in_circle = 0
        while not self.converged:
            total_thrown += darts_per_round
            in_circle += self.throw_and_score(darts_per_round)
            pi_approx = 4 * in_circle/total_thrown
            self.history = pi_approx

        return round(self.history[-1], self.significant_digits - 1)
