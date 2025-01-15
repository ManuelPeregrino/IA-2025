import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os

# Parameters
min_val = int(input("Ingresa el valor minimo: "))
max_val = int(input("Ingresa el valor maximo: "))
population_size = int(input("Ingresa el tamaño maximo de la poblacion: "))
delta_x = float(input("Ingresa el valor de delta x (precisión): "))
generations = int(input("Ingresa el numero de generaciones: "))
mutation_rate = float(input("Ingresa la probabilidad de mutacion (usa numeros flotantes): "))

# Create a directory to store the images
if not os.path.exists('generation_images'):
    os.makedirs('generation_images')

def populate(population_size, min_val, max_val, delta_x):
    """Generate an initial population of random floating-point numbers."""
    possible_values = np.arange(min_val, max_val + delta_x, delta_x)
    return [random.choice(possible_values) for _ in range(population_size)]

def fitness_function(x):
    """Example fitness function to maximize."""
    return 0.1 * x * np.log(1 * abs(x) + 1e-10) * np.cos(x) ** 2

def evaluate_population(population):
    """Evaluate the fitness of each individual in the population."""
    return [fitness_function(individual) for individual in population]

def select_parents(population, fitness):
    """Select two parents using roulette wheel selection."""
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]

    total_fitness = sum(fitness)
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
    
    best_fitness_per_generation = []
    worst_fitness_per_generation = []
    mean_fitness_per_generation = []

    # Plot the fitness function as background
    x_vals = np.linspace(min_val-5, max_val+5, 500)
    y_vals = fitness_function(x_vals)
    
    for generation in range(generations):
        fitness = evaluate_population(population)

        best_fitness = max(fitness)
        worst_fitness = min(fitness)
        mean_fitness = np.mean(fitness)

        # Store the results for plotting later
        best_fitness_per_generation.append(best_fitness)
        worst_fitness_per_generation.append(worst_fitness)
        mean_fitness_per_generation.append(mean_fitness)

        # Log the best fitness in the current generation
        best_individual = population[fitness.index(best_fitness)]
        print(f"Generation {generation}: Best fitness = {best_fitness}, Worst fitness = {worst_fitness}, Mean fitness = {mean_fitness}")

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, delta_x, mutation_rate)
            child2 = mutate(child2, delta_x, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

        # Create and save the plot for the current generation
        plt.figure(figsize=(10, 6))

        # Plot the fitness function as background
        plt.plot(x_vals, y_vals, label='Fitness Function', color='gray', alpha=0.4)
        
        # Plot the best and worst fitness points
        plt.scatter(best_individual, best_fitness, color='green', label='Best Fitness', zorder=5, marker='o')  # Green dot for best
        plt.scatter(population[fitness.index(worst_fitness)], worst_fitness, color='red', label='Worst Fitness', zorder=5, marker='D')  # Red diamond for worst

        # Plot the evolution lines
        plt.plot(range(generation + 1), best_fitness_per_generation, label='Best Fitness', color='green')
        plt.plot(range(generation + 1), worst_fitness_per_generation, label='Worst Fitness', color='red')
        plt.plot(range(generation + 1), mean_fitness_per_generation, label='Mean Fitness', color='blue', linestyle='--')

        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Evolution at Generation {generation}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an image
        plt.savefig(f'generation_images/gen_{generation}.png')
        plt.close()

    # Final results
    fitness = evaluate_population(population)
    best_fitness = max(fitness)
    best_individual = population[fitness.index(best_fitness)]
    print(f"Best individual: {best_individual} => Fitness: {best_fitness}")
    
    # Create a video from the saved images
    frame_array = []
    for generation in range(generations):
        img = cv2.imread(f'generation_images/gen_{generation}.png')
        frame_array.append(img)

    # Video parameters
    height, width, layers = frame_array[0].shape
    video_name = 'genetic_algorithm_evolution.avi'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for frame in frame_array:
        out.write(frame)
    
    out.release()

    # Clean up image files
    for generation in range(generations):
        os.remove(f'generation_images/gen_{generation}.png')

    print(f"Video saved as {video_name}")

    return best_individual, best_fitness

# Run the genetic algorithm
genetic_algorithm(min_val, max_val, population_size, delta_x, generations, mutation_rate)
