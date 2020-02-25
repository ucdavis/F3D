import pytest
from pi_calculator_oop import pi_calculator


@pytest.mark.parametrize(
    "sig_digs, expected",
    [
        (1, 3),
        (3, 3.14)
    ]
)
def test__PiCalculator(sig_digs, expected):
    for n_darts_per_throw in [100, 200, 300]:
        calculator = pi_calculator.PiCalculator(n_darts_per_throw)
        assert isinstance(calculator, pi_calculator.PiCalculator)
        assert calculator.calculate(sig_digs) == expected
