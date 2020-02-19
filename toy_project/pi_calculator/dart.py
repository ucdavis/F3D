import numpy as np


class Dart:
    """
    A thrown dart.

    Attributes
    ----------
    coords: np.array
        The x,y coordinates of the dart on the square.
    radius: float
        The norm from the origin.
    in_circle: bool
        Whether the dart fell in the unit circle.
    """
    def __init__(self, coords):
        self.coords = coords

    @property
    def radius(self):
        return np.linalg.norm(self.coords)

    @property
    def in_circle(self):
        return self.radius <= 1