This folder holds several examples of a toy project to calculate pi through
counting points that fall into the unit circle compared to the outer tangent
square.


## Description
There are three distinct implementations, following three coding paradigms supported
by python (Procedural, Functional, and Object-Oriented programming). As you go down 
the list of these implementations, the coding style becomes more "advanced", meaning 
written in a way you may have less exposure to. However, as you go down this list, the 
"cleanliness" of the implementation also increases--there is progressively more logical
grouping, and cleaner separations of concerns, which makes adding new features (or changing
existing ones) easier to implement while remaining confident the code works. 

### Procedural Implementation
First, you'll find a procedural (script) implementation in `pi_calculator_procedural`.
This is a single script called ``calculate_pi.py``, which is called via 

```bash
python calculate_pi.py <n_digits> <n_darts_per_scoring>
```
This will print the result to standard out. Note that since everything is in a single
script, there are no natural groupings in the code, aside from those created by comment
decorations (sometimes called "ASCII art"). This makes it hard to test.

### Functional Implementation
A functional implementation is in `pi_calculator_functional`. You'll notice that functional
implementations are much more descriptive; the input and output of the functions (in parlance
sometimes mistakenly referred to as the API) give a clear contract of what's getting ingested
and spit back out by each unit of work. In addition, since the code is now split into units,
we can write unit tests of these functions. You'll find such tests in `test__calculate_pi.py`
in the style of `pytest`.

### Object-Oriented Implementation
The functional implementation is a big improvement (and for some things is definitely the most preferred).
However, there is some bundling of concerns. If you read through the functions, you'll notice some
functions have to do with scoring / calculating `pi`, while others have to do with simulating the
mechanics of throwing darts. In object-oriented programming (OOP), we'd collect these functions
as methods of `PiCalculator` and `Dart` objects, and use the notion of class properties to store
variables we might need. This saves us some annoying passing of variables through functions, which
makes refactoring difficult.

### Performance
This is obviously not the fastest way to calculate `pi`. However, each implementation has its own
performance characteristics.

The procedural implementation and functional implementations are faster on a single calculation.
However, the OOP preserves the history of the dart throws, and therefore is much quicker in successive
calculations. The reason for the speed discrepancy is that classes (objects) in python are extremely
flexible, and therefore hard to compile efficiently. 

## "Installation"
To install the required packages, ensure you have `pip` installed, and do

```bash
pip install -r requirements.txt
pip install .
```
inside the `examples` directory. This will install all three folders as packages you can then
import into your python prompt, e.g.

```python
import pi_calculator_procedural as pi_calculator
```

## Usage
The functional and OOP calculators have python APIs. Use the OOP implementation with
```python
from pi_calculator_oop import pi_calculator
# initialize the calculator by specify the desired precision
calculator = pi_calculator.PiCalculator(darts_per_score=100)
# execute the calculation.
pi = calculator.calculate(significant_digits=3)  # should return 3.14
```
and the functional calculator with
```python
import pi_calculator_functional as pi_calculator
pi_calculator.main(significant_digits=3, n_darts_per_scoring=100)

```


This is not a very efficient method for calculating `pi`, so be very patient
if you set `significant_digits` above `4`.

