This folder represents toy project and continuous integration workflow, 
to use as an example of a modular, tested codebase (and of setting up 
simple continuous integration through github actions).

## Description
Creates a toy project that calculates pi through counting the number of 
points that fall in the unit circle compared to the outer tangent square.

The folder toy_project contains the codes used in the calculation, 
packaged in a python package called pi_calculator. It can be installed 
via pip install /path/to/toy_project, and imported via import pi_calculator.

The tests directory contains unit tests of utility functions and the helper 
Dart class, as well as an end-to-end test of the calculator functionality.

There is a github action workflow checked in to .github/workflows titled 
toy-project.yaml that executes the tests on commit against a virtual ubuntu 
machine for continuous integration.

## "Installation"
To install the required packages, ensure you have `pip` installed, and do

```bash
pip install -r requirements.txt
pip install .
```
inside the `toy_project` directory.

## Usage
The calculator has a python API. Use with
```python
import pi_calculator
# initialize the calculator by specify the desired precision
calculator = pi_calculator.PiCalculator(significant_digits=3)
# execute the calculation.
pi = calculator()
```
This is not a very efficient method for calculating `pi`, so be very patient
if you set `significant_digits` above `4`.

