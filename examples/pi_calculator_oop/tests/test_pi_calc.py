import pytest
from pi_calculator_oop import calculate_pi


@pytest.mark.parametrize(
    "sig_digs, expected",
    [
        (1, 3),
        (2, 3.1),
        (3, 3.14)
    ]
)
def test__PiCalculator(sig_digs, expected):
    for n_darts_per_throw in [1, 3, 10]:
        calculator = calculate_pi.PiCalculator(n_darts_per_throw)
        assert isinstance(calculator, calculate_pi.PiCalculator)
        assert calculator.calculate(sig_digs) == expected
