import pytest
from pi_calculator_oop import pi_calc


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
        calculator = pi_calc.PiCalculator(n_darts_per_throw)
        assert isinstance(calculator, pi_calc.PiCalculator)
        assert calculator.calculate(sig_digs) == expected
