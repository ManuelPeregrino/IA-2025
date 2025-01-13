#Manuel Alejandro Peregrino Clemente

import numpy as np
import random

# Parameters
min_val = 5
max_val = 10
population_size = int(input("Ingresa el tamaño maximo de la poblacion:"))
bit_length = int(input("Ingresa el tamaño maximo de bits: "))
generations = int(input("Ingresa el numero de generaciones: "))
mutation_rate = float(input("Ingresa la probabilidad de mutacion (usa numeros flotantes): "))


def populate(population_size, bit_length):
    """Generate an initial population of random binary strings."""
    return [''.join(random.choice('01') for _ in range(bit_length)) for _ in range(population_size)]


def bits_to_number(bits, min_val, max_val):
    """Convert binary string to a real number within a given range."""
    decimal = int(bits, 2)
    max_decimal = 2**len(bits) - 1
    return min_val + (max_val - min_val) * decimal / max_decimal


def fitness_function(x):
    """Example fitness function to maximize: f(x) = -x^2 + 10x."""
    return 0.1 * x * np.log(1 * abs(x) + 1e-10) * np.cos(x) ** 2
#The 1e-10 helps avoiding a division by 0

def evaluate_population(population, min_val, max_val):
    """Evaluate the fitness of each individual in the population."""
    return [fitness_function(bits_to_number(individual, min_val, max_val)) for individual in population]

def select_parents(population, fitness):
    """Select two parents using roulette wheel selection."""
    # Shift fitness values to be non-negative
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]
    
    total_fitness = sum(fitness)
    # Handle cases where total_fitness might still be 0
    if total_fitness == 0:
        probabilities = [1 / len(fitness) for _ in fitness]
    else:
        probabilities = [f / total_fitness for f in fitness]
    
    parents = np.random.choice(population, size=2, p=probabilities)
    return parents

def crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    """Perform mutation on an individual with a given mutation rate."""
    mutated = ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit))
        for bit in individual
    )
    return mutated

def genetic_algorithm(
    min_val, max_val, population_size, bit_length, generations, mutation_rate
):
    """Run the genetic algorithm to maximize the fitness function."""
    population = populate(population_size, bit_length)

    for generation in range(generations):
        fitness = evaluate_population(population, min_val, max_val)

        # Log the best fitness in the current generation
        best_fitness = max(fitness)
        best_individual = population[fitness.index(best_fitness)]
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final results
    fitness = evaluate_population(population, min_val, max_val)
    best_fitness = max(fitness)
    best_individual = population[fitness.index(best_fitness)]
    best_value = bits_to_number(best_individual, min_val, max_val)
    print(f"Best individual: {best_individual} => Value: {best_value}, Fitness: {best_fitness}")
    return best_individual, best_value, best_fitness


# Run the genetic algorithm
genetic_algorithm(min_val, max_val, population_size, bit_length, generations, mutation_rate)