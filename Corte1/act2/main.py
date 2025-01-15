import numpy as np
import random

# Parameters
min_val = int(input("Ingresa el valor minimo"))
max_val = int(input("Ingresa el valor maximo"))
population_size = int(input("Ingresa el tamaño maximo de la poblacion:"))
delta_x = float(input("Ingresa el valor de delta x (precisión): "))
generations = int(input("Ingresa el numero de generaciones: "))
mutation_rate = float(input("Ingresa la probabilidad de mutacion (usa numeros flotantes): "))

def populate(population_size, min_val, max_val, delta_x):
    """Generate an initial population of random floating-point numbers."""
    possible_values = np.arange(min_val, max_val + delta_x, delta_x)
    return [random.choice(possible_values) for _ in range(population_size)]

def fitness_function(x):
    """Example fitness function to maximize."""
    return 0.1 * x * np.log(1 * abs(x) + 1e-10) * np.cos(x) ** 2

# The 1e-10 helps avoiding a division by 0
def evaluate_population(population):
    """Evaluate the fitness of each individual in the population."""
    return [fitness_function(individual) for individual in population]

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
    """Perform arithmetic crossover between two parents."""
    child1 = (parent1 + parent2) / 2
    child2 = (parent1 - parent2) / 2 + parent1  # Another combination
    return child1, child2

def mutate(individual, delta_x, mutation_rate):
    if random.random() < mutation_rate:
        mutation = random.uniform(-delta_x, delta_x)
        mutated = individual + mutation
        return max(min(mutated, max_val), min_val)  # Restringir al rango permitido
    return individual


def genetic_algorithm(min_val, max_val, population_size, delta_x, generations, mutation_rate):
    """Run the genetic algorithm to maximize the fitness function."""
    population = populate(population_size, min_val, max_val, delta_x)

    for generation in range(generations):
        fitness = evaluate_population(population)

        # Log the best fitness in the current generation
        best_fitness = max(fitness)
        best_individual = population[fitness.index(best_fitness)]
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, delta_x, mutation_rate)
            child2 = mutate(child2, delta_x, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final results
    fitness = evaluate_population(population)
    best_fitness = max(fitness)
    best_individual = population[fitness.index(best_fitness)]
    print(f"Best individual: {best_individual} => Fitness: {best_fitness}")
    return best_individual, best_fitness

# Run the genetic algorithm
genetic_algorithm(min_val, max_val, population_size, delta_x, generations, mutation_rate)
