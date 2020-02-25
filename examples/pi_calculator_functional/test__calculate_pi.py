import pytest
import numpy as np
from .calculate_pi import *


def test__main(capsys):
    main(3, 100)
    captured_stdout = capsys.readouterr()
    assert captured_stdout.out == "3.14"


@pytest.mark.parametrize(
    "sig_digs, expected",
    [
        (1, 3),
        (2, 3.1),
        (3, 3.14)
    ]
)
def test__calculate_pi(sig_digs, expected):
    estimate = calculate_pi(sig_digs)
    assert estimate == expected


def test__get_threshold():
    for sigdigs in [1, 3, 10]:
        assert get_threshold(sigdigs) == 10**(-sigdigs)


def test__is_converged():
    for threshold in np.logspace(-10, -3, 5):
        history = np.zeros(100)
        assert is_converged(threshold, history)
        history = np.arange(100)
        assert not is_converged(threshold, history)


def test__get_number_in_circle():
    for thrown in [0, 10, 100, 1000]:
        assert get_number_in_circle(thrown) <= thrown
        assert get_number_in_circle(thrown) >= 0


def get_random_coords():
    for thrown in [0, 10, 100, 1000]:
        coords = get_random_coords(thrown)
        assert coords.shape() == (thrown, 2)
        assert np.all(coords <= 1)
        assert np.all(coords >= -1)
