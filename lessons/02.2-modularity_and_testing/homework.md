1. Read [Adam's second notebook](https://github.com/leknox/Scientific-Python-Tutorials/blob/master/02%20Functions%20and%20Classes.ipynb) where he introduces functions, classes, and unit tests.

2. Write a code to generate and display a cellular automaton with two changes from the binary CA case:

    1. replace the 2-state system with a 3-state system. Thus a cell, instead of being either 0 or 1, can be 0, 1, or 2.

    2. Note: this is the original assignment, but it doesn't actually lead to a great example of code reuse: define the neighborhood as the cell above and to the left, rather than above, to the left, and to the right. This is so that there are 3^9 = 19683 neighborhoods instead of 3^(3^3) which is greater than 10^{12}. For next time I recommend doing the full 3 point neighborhood 3-state system. The coding isn't actually that much more and the reusability is better.

    3. Do two versions: one without use of functions, and one with use of functions and at least one unit test. See if you can reuse some of Adam Rupe's 2-state CA code. 