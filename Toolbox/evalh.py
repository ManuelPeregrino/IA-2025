import numpy as np
import random

# Parameters
min_val = -300  # Minimum value in the range
max_val = 200   # Maximum value in the range
delta_x = 0.01  # Step size for granularity
population_size = int(input("Enter the maximum population size: "))
bit_length = int(input("Enter the bit length: "))
generations = int(input("Enter the number of generations: "))
mutation_rate = float(input("Enter the mutation probability (use floating-point numbers): "))
p_crossover = float(input("Enter the crossover probability (use floating-point numbers): "))

def populate(population_size, bit_length):
    """Create an initial population of binary strings."""
    return [''.join(random.choice('01') for _ in range(bit_length)) for _ in range(population_size)]

def bits_to_number(bits, min_val, max_val):
    """Convert binary string to a real number within a given range."""
    decimal = int(bits, 2)
    max_decimal = 2**len(bits) - 1
    return min_val + (max_val - min_val) * decimal / max_decimal

def fitness_function(x):
    """Calculate the fitness function f(x) = ln(|x^3|) * cos(x) * sin(x)."""
    if x == 0:
        return -np.inf  # Avoid log(0)
    return np.log(abs(x**3)) * np.cos(x) * np.sin(x)

def evaluate_population(population, min_val, max_val):
    """Evaluate the fitness of each individual in the population."""
    return [fitness_function(bits_to_number(individual, min_val, max_val)) for individual in population]

def form_pairs(population, fitness):
    """Form pairs based on the strategy: The best third dominates."""
    sorted_population = [x for _, x in sorted(zip(fitness, population), reverse=True)]
    best, middle, worst = np.array_split(sorted_population, 3)
    
    pairs = []
    for group, allowed_groups in [(best, [population]),
                                  (middle, [middle, worst]),
                                  (worst, [worst])]:
        for individual in group:
            if random.random() < p_crossover:
                partner_group = random.choice(allowed_groups)
                partner = random.choice(partner_group)
                pairs.append((individual, partner))
    return pairs

def multi_point_crossover(parent1, parent2):
    """Perform crossover with random points."""
    k = random.randint(1, len(parent1) // 2)
    points = sorted(random.sample(range(1, len(parent1)), k))
    child1, child2 = parent1, parent2
    for i in range(len(points)):
        if i % 2 == 0:
            child1 = child1[:points[i]] + parent2[points[i]:]
            child2 = child2[:points[i]] + parent1[points[i]:]
    return child1, child2

def mutate(individual, mutation_rate):
    """Apply the sum and modulo mutation strategy."""
    if random.random() < mutation_rate:
        s = random.randint(1, 2**(len(individual) // 2))
        x = bits_to_number(individual, min_val, max_val)
        new_x = (x + s) % (max_val - min_val)
        return format(int((new_x - min_val) / (max_val - min_val) * (2**len(individual) - 1)), f'0{len(individual)}b')
    return individual

def prune_population(population, fitness, max_size):
    """Prune the population equitably by groups."""
    unique_population = list({ind: None for ind in population}.keys())
    sorted_population = [x for _, x in sorted(zip(fitness, unique_population), reverse=True)]
    best = sorted_population[:1]
    remaining = sorted_population[1:]
    split_index = len(remaining) // 2
    best_group = remaining[:split_index]
    worst_group = remaining[split_index:]
    excess = len(sorted_population) - max_size
    if excess > 0:
        best_to_remove = min(len(best_group), excess // 2)
        worst_to_remove = min(len(worst_group), excess - best_to_remove)
        for _ in range(best_to_remove):
            if best_group:
                best_group.pop(random.randint(0, len(best_group) - 1))
        for _ in range(worst_to_remove):
            if worst_group:
                worst_group.pop(random.randint(0, len(worst_group) - 1))
    return best + best_group + worst_group

def genetic_algorithm(min_val, max_val, population_size, bit_length, generations, mutation_rate, p_crossover):
    """Run the genetic algorithm."""
    population = populate(population_size, bit_length)
    for generation in range(generations):
        fitness = evaluate_population(population, min_val, max_val)
        best_fitness = max(fitness)
        print(f"Generation {generation}: Best fitness = {best_fitness}")
        pairs = form_pairs(population, fitness)
        new_population = []
        for parent1, parent2 in pairs:
            child1, child2 = multi_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        fitness = evaluate_population(new_population, min_val, max_val)
        population = prune_population(new_population, fitness, population_size)
    final_fitness = evaluate_population(population, min_val, max_val)
    best_fitness = max(final_fitness)
    best_individual = population[final_fitness.index(best_fitness)]
    best_value = bits_to_number(best_individual, min_val, max_val)
    print(f"Best individual: {best_individual} => Value: {best_value}, Fitness: {best_fitness}")
    return best_individual, best_value, best_fitness

# Run the genetic algorithm
genetic_algorithm(min_val, max_val, population_size, bit_length, generations, mutation_rate, p_crossover)
