import two_state_ca
from matplotlib import pyplot as plt

initial_conditions = two_state_ca.random_string(10)
field = two_state_ca.spacetime_field(110, initial_conditions, 10)
two_state_ca.spacetime_diagram(field)
plt.savefig('rule110.pdf')