

import population_tools as ptool
import numpy as np


population = [] ##Test array. Remove after testing
test_population_size = 10 ##Test value, remove after testing
test_dataset = "raw_data.csv"


population = ptool.fill_population_with_csv(test_dataset, population)
individual = ptool.get_dataset_target_value(population, 'x')
print(individual)

def fitness_function(x):
    """Example fitness function to maximize."""
    return 0.1 * x * np.log(1 * abs(x) + 1e-10) * np.cos(x) ** 2

