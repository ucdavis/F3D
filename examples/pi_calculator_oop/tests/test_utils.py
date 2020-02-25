import numpy as np
import pytest
from pi_calculator_oop import utils


@pytest.mark.parametrize("num_pairs", [0, 1, 10, 30])
def test__get_random_pairs(num_pairs):
    pairs = utils._get_random_pairs(num_pairs)
    assert isinstance(pairs, np.ndarray)
    assert len(pairs.shape) == 2
    assert pairs.shape[0] == num_pairs
    assert pairs.shape[1] == 2
    assert np.all(pairs < 1)
    assert np.all(-1 < pairs)


def test__get_random_pair_bad_args():
    with pytest.raises(Exception):
        utils._get_random_pairs(1.23)

    with pytest.raises(Exception):
        utils._get_random_pairs(-2)

    with pytest.raises(Exception):
        utils._get_random_pairs([2, 4])

    with pytest.raises(Exception):
        utils._get_random_pairs(np.array([2, 4]))
