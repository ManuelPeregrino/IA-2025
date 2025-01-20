#Manuel Alejandro Peregrino Clemente
from individual import Individual
import random
import csv

#This method helps us filling/initializing our population with random numbers under a given range
def fill_population_with_random_numbers(population, population_size,population_range_min, population_range_max):
    for _ in range(population_size):
        population.append(random.randint(population_range_min, population_range_max))
    return population

#This method handles a CSV file and converts it into a python object.
def fill_population_with_csv(csv_path, population):
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        population = [Individual(row) for row in csv_reader]
    return population

#This method will return a filtered version of the dataset if needed
def get_dataset_target_value(population, target_value):
    returned_values = [{key: obj[key] for key in vars(obj) if key.startswith(target_value)} for obj in population]
    return returned_values

