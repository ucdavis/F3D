import pytest
from pi_calculator import pi_calc


@pytest.mark.parametrize(
    "sig_digs, expected",
    [
        (1, 3),
        (2, 3.1),
        (3, 3.14)
    ]
)
def test__PiCalculator(sig_digs, expected):
    calculator = pi_calc.PiCalculator(significant_digits=sig_digs)
    assert isinstance(calculator, pi_calc.PiCalculator)
    assert calculator() == expected
