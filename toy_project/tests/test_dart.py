import numpy as np
from pi_calculator import Dart


def test_Dart():
    dart = Dart(np.array([0.0, 0.0]))
    assert dart.radius == 0.0
    assert dart.in_circle

    dart = Dart(np.array([0.9,0.9]))
    assert not dart.in_circle
