import random
from matplotlib import pyplot as plt

def random_string(length):
    '''
    Returns a random bit string of the given length. 
    
    Parameters
    ----------
    length: int
        Posivite integer that specifies the desired length of the bit string.
        
    Returns
    -------
    out: list
        The random bit string given as a list, with int elements.
    '''
    if not isinstance(length, int) or length < 0:
        raise ValueError("input length must be a positive ingeter")
    return [random.randint(0,1) for _ in range(length)]

def neighborhoods():
    '''
    Returns a list of neighborhood tuples in lexicographical order.
    '''
    return [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

def lookup_table(rule_number):
    '''
    Returns a dictionary which maps ECA neighborhoods to output values. 
    Uses Wolfram rule number convention.
    
    Parameters
    ----------
    rule_number: int
        Integer value between 0 and 255, inclusive. Specifies the ECA lookup table
        according to the Wolfram numbering scheme.
        
    Returns
    -------
    lookup_table: dict
        Lookup table dictionary that maps neighborhood tuples to their output according to the 
        ECA local evolution rule (i.e. the lookup table), as specified by the rule number. 
    '''
    if not isinstance(rule_number, int) or rule_number < 0 or rule_number > 255:
        raise ValueError("rule_number must be an int between 0 and 255, inclusive")
    neighborhoods = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
    in_binary = '{:{fill}{align}{width}b}'.format(rule_number, 
                                                  fill='0', 
                                                  align='>', 
                                                  width='8')
    
    return dict(zip(neighborhoods, map(int,reversed(in_binary)))) # use map so that outputs are ints, not strings

def spacetime_field(rule_number, initial_condition, time_steps):
    '''
    Returns a spacetime field array using the given rule number on the 
    given initial condition for the given number of time steps.
    
    Parameters
    ----------
    rule_number: int
        Integer value between 0 and 255, inclusive. Specifies the ECA lookup table
        according to the Wolfram numbering scheme.
    initial_condition: list
        Binary string used as the initial condition for the ECA. Elements of the list
        should be ints. 
    time_steps: int
        Positive integer specifying the number of time steps for evolving the ECA. 
    '''
    if time_steps < 0:
        raise ValueError("time_steps must be a non-negative integer")
    # try converting time_steps to int and raise a custom error if this can't be done
    try:
        time_steps = int(time_steps)
    except ValueError:
        raise ValueError("time_steps must be a non-negative integer")
        
    # we will see a cleaner and more efficient way to do the following when we introduce numpy
    for i in initial_condition:
        if i not in [0,1]:
            raise ValueError("initial condition must be a list of 0s and 1s")
        
    lookup = lookup_table(rule_number)
    length = len(initial_condition)
    
    # initialize spacetime field and current configuration
    spacetime_field = [initial_condition]
    current_configuration = initial_condition.copy()

    # apply the lookup table to evolve the CA for the given number of time steps
    for t in range(time_steps):
        new_configuration = []
        for i in range(length):

            neighborhood = (current_configuration[(i-1)], 
                            current_configuration[i], 
                            current_configuration[(i+1)%length])

            new_configuration.append(lookup[neighborhood])

        current_configuration = new_configuration
        spacetime_field.append(new_configuration)
    
    return spacetime_field

def spacetime_diagram(spacetime_field, size=12, colors=plt.cm.Greys):
    '''
    Produces a simple spacetime diagram image using matplotlib imshow with 'nearest' interpolation.
    
   Parameters
    ---------
    spacetime_field: array-like (2D)
        1+1 dimensional spacetime field, given as a 2D array or list of lists. Time should be dimension 0;
        so that spacetime_field[t] is the spatial configuration at time t. 
        
    size: int, optional (default=12)
        Sets the size of the figure: figsize=(size,size)
    colors: matplotlib colormap, optional (default=plt.cm.Greys)
        See https://matplotlib.org/tutorials/colors/colormaps.html for colormap choices.
        A colormap 'cmap' is called as: colors=plt.cm.cmap
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(spacetime_field, cmap=colors, interpolation='nearest')
    plt.show()