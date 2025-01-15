import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2
import os
import threading  # Import threading to run the algorithm in a separate thread

# Create a directory to store the images
if not os.path.exists('generation_images'):
    os.makedirs('generation_images')

def fitness_function(x):
    """Example fitness function to maximize with a cap applied element-wise."""
    fitness = 0.1 * x * np.log(1 * np.abs(x) + 1e-10) * np.cos(x) ** 2
    fitness = np.minimum(fitness, 100)  # Apply the cap element-wise
    return fitness


def populate(population_size, min_val, max_val, delta_x):
    """Generate an initial population of random floating-point numbers."""
    possible_values = np.arange(min_val, max_val + delta_x, delta_x)
    return [random.choice(possible_values) for _ in range(population_size)]

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

def mutate(individual, min_val, max_val, delta_x, mutation_rate):
    """Apply mutation to an individual with a controlled step size."""
    if random.random() < mutation_rate:
        # Ensure mutation step is within reasonable bounds (proportional to delta_x)
        mutation_step = random.uniform(-delta_x, delta_x)
        mutated = individual + mutation_step

        # Ensure the mutated individual is within bounds
        mutated = max(min(mutated, max_val), min_val)
        
        # Apply the boundary cap to prevent runaway mutation values
        return mutated
    return individual



def prune_population(population, fitness, population_size):
    """Prune the population by removing the worst individuals."""
    sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
    # Remove worst individuals
    pruned_population = [individual for individual, _ in sorted_population[:population_size]]
    return pruned_population

def genetic_algorithm(min_val, max_val, population_size, delta_x, generations, mutation_rate, plot_canvas, fig):
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

        # Create a new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, min_val, max_val, delta_x, mutation_rate)
            child2 = mutate(child2, min_val, max_val, delta_x, mutation_rate)
            new_population.extend([child1, child2])

        # Ensure the new population does not exceed the maximum population size
        population = prune_population(new_population, evaluate_population(new_population), population_size)

        # Clear the previous plot and start fresh
        ax = fig.add_subplot(111)

        # Plot the fitness function as background
        ax.plot(x_vals, y_vals, label='Fitness Function', color='gray', alpha=0.4)
        
        # Plot the best and worst fitness points
        ax.scatter(population[fitness.index(best_fitness)], best_fitness, color='green', label='Best Fitness', zorder=5, marker='o')  # Green dot for best
        ax.scatter(population[fitness.index(worst_fitness)], worst_fitness, color='red', label='Worst Fitness', zorder=5, marker='D')  # Red diamond for worst

        # Plot the evolution lines
        ax.plot(range(generation + 1), best_fitness_per_generation, label='Best Fitness', color='green')
        ax.plot(range(generation + 1), worst_fitness_per_generation, label='Worst Fitness', color='red')
        ax.plot(range(generation + 1), mean_fitness_per_generation, label='Mean Fitness', color='blue', linestyle='--')

        # Refresh labels and legend
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.set_title(f'Fitness Evolution at Generation {generation}')
        ax.legend(loc='upper left')
        ax.grid(True)

        # Redraw the canvas
        plot_canvas.draw()
        ax.clear()  # Clear the previous plot to avoid overlap

    # Final results
    fitness = evaluate_population(population)
    best_fitness = max(fitness)
    best_individual = population[fitness.index(best_fitness)]
    print(f"Best individual: {best_individual} => Fitness: {best_fitness}")

    return best_individual, best_fitness

def start_algorithm():
    min_val = int(min_val_entry.get())
    max_val = int(max_val_entry.get())
    population_size = int(population_size_entry.get())
    delta_x = float(delta_x_entry.get())
    generations = int(generations_entry.get())
    mutation_rate = float(mutation_rate_entry.get())

    # Create the figure for plotting
    fig = plt.Figure(figsize=(10, 6), dpi=100)
    
    # Create a canvas to display the plot in the Tkinter window
    plot_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Start the genetic algorithm in a separate thread to avoid blocking the Tkinter main loop
    threading.Thread(target=genetic_algorithm, args=(min_val, max_val, population_size, delta_x, generations, mutation_rate, plot_canvas, fig), daemon=True).start()

# Create Tkinter root window
root = tk.Tk()
root.title("Genetic Algorithm Visualization")

# Set window size
root.geometry("1440x900")

# Input fields
input_frame = tk.Frame(root)
input_frame.pack(pady=20)

# Min and Max values
min_val_label = tk.Label(input_frame, text="Min Value:")
min_val_label.grid(row=0, column=0, padx=10)
min_val_entry = tk.Entry(input_frame)
min_val_entry.grid(row=0, column=1)

max_val_label = tk.Label(input_frame, text="Max Value:")
max_val_label.grid(row=0, column=2, padx=10)
max_val_entry = tk.Entry(input_frame)
max_val_entry.grid(row=0, column=3)

# Population size
population_size_label = tk.Label(input_frame, text="Population Size:")
population_size_label.grid(row=1, column=0, padx=10)
population_size_entry = tk.Entry(input_frame)
population_size_entry.grid(row=1, column=1)

# Delta x
delta_x_label = tk.Label(input_frame, text="Delta X:")
delta_x_label.grid(row=1, column=2, padx=10)
delta_x_entry = tk.Entry(input_frame)
delta_x_entry.grid(row=1, column=3)

# Generations
generations_label = tk.Label(input_frame, text="Generations:")
generations_label.grid(row=2, column=0, padx=10)
generations_entry = tk.Entry(input_frame)
generations_entry.grid(row=2, column=1)

# Mutation rate
mutation_rate_label = tk.Label(input_frame, text="Mutation Rate:")
mutation_rate_label.grid(row=2, column=2, padx=10)
mutation_rate_entry = tk.Entry(input_frame)
mutation_rate_entry.grid(row=2, column=3)

# Start button
start_button = tk.Button(root, text="Start Algorithm", command=start_algorithm)
start_button.pack(pady=20)

# Frame for the plot
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Run the Tkinter event loop
root.mainloop()
