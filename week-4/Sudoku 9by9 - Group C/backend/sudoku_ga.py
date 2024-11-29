import random
from copy import deepcopy

def initialize_population(initial_board, population_size):
    population = []
    fixed_cells = (initial_board != 0)
    for _ in range(population_size):
        individual = deepcopy(initial_board)
        for i in range(9):
            missing = list(set(range(1, 10)) - set(individual[i, :]))
            empty_indices = [j for j in range(9) if not fixed_cells[i, j]]
            random.shuffle(missing)
            for j in empty_indices:
                individual[i, j] = missing.pop() if missing else random.randint(1, 9)
        population.append(individual)
    return population

def fitness(individual):
    fitness_score = 0
    for i in range(9):
        fitness_score += len(set(individual[i, :]))
    for j in range(9):
        fitness_score += len(set(individual[:, j]))
    for bi in range(3):
        for bj in range(3):
            block = individual[bi * 3:(bi + 1) * 3, bj * 3:(bj + 1) * 3].flatten()
            fitness_score += len(set(block))
    return fitness_score

def tournament_selection(population, population_fitness):
    selected = []
    population_size = len(population)
    for _ in range(population_size):
        i, j = random.sample(range(population_size), 2)
        if population_fitness[i] > population_fitness[j]:
            selected.append(deepcopy(population[i]))
        else:
            selected.append(deepcopy(population[j]))
    return selected

def crossover(parent1, parent2, fixed_cells, initial_board):
    child = deepcopy(parent1)
    crossover_point = random.randint(1, 8)
    child[crossover_point:, :] = parent2[crossover_point:, :]
    child[fixed_cells] = initial_board[fixed_cells]
    return child

def mutate(individual, fixed_cells, mutation_rate):
    for i in range(9):
        if random.random() < mutation_rate:
            indices = [j for j in range(9) if not fixed_cells[i, j]]
            if len(indices) >= 2:
                a, b = random.sample(indices, 2)
                individual[i, a], individual[i, b] = individual[i, b], individual[i, a]
    return individual

def run_genetic_algorithm(initial_board, population_size, generations, mutation_rate, max_no_improvement):
    fixed_cells = (initial_board != 0)
    population = initialize_population(initial_board, population_size)
    best_solution = None
    max_fitness = 0
    graph_data = []
    no_improvement_generations = 0
    history = []

    for generation in range(generations):
        population_fitness = [fitness(ind) for ind in population]
        sorted_indices = sorted(range(len(population_fitness)), key=lambda k: population_fitness[k], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [population_fitness[i] for i in sorted_indices]

        graph_data.append(sorted_fitness[0])

        if sorted_fitness[0] > max_fitness:
            max_fitness = sorted_fitness[0]
            best_solution = {"generation": int(generation), "fitness": int(max_fitness),
                "board": sorted_population[0].astype(int).tolist(), "is_complete": False}
            no_improvement_generations = 0
            history.append(best_solution)
        else:
            no_improvement_generations += 1

        if no_improvement_generations >= max_no_improvement:
            best_solution["message"] = f"No improvement after ${max_no_improvement} generations."
            history.append(best_solution)
            return history, graph_data

        if max_fitness == 243:
            best_solution["message"] = "Solution found."
            best_solution["is_complete"] = True
            history.append(best_solution)
            return history, graph_data

        selected = tournament_selection(sorted_population, sorted_fitness)

        next_generation = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            child1 = crossover(parent1, parent2, fixed_cells, initial_board)
            child2 = crossover(parent2, parent1, fixed_cells, initial_board)
            next_generation.extend([child1, child2])

        population = [mutate(ind, fixed_cells, mutation_rate) for ind in next_generation]
        print(f"Generation {generation} - Best fitness: {sorted_fitness[0]}")

    if best_solution:
        best_solution["message"] = "GA terminated without finding a complete solution."
        history.append(best_solution)
    return history, graph_data