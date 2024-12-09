import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# CORS middleware for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to initialize a random population
def initialize_population(population_size, board_size):
    population = []
    for _ in range(population_size):
        chromosome = random.sample(range(board_size * board_size), board_size)
        population.append(chromosome)
    return population

def fitness(chromosome, board_size):
    """Calculate fitness as the number of non-attacking bishops."""
    score = 0
    for i in range(len(chromosome)):
        valid = True
        x1, y1 = divmod(chromosome[i], board_size)
        for j in range(len(chromosome)):
            if i != j:
                x2, y2 = divmod(chromosome[j], board_size)
                if abs(x1 - x2) == abs(y1 - y2):  # Diagonal threat
                    valid = False
                    break
        if valid:
            score += 1
    return score

# Single-point crossover to produce two children
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return [child1, child2]


# Mutation function to change a random position in the chromosome
def mutate(chromosome, mutation_rate, board_size):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(chromosome) - 1)
        new_pos = random.randint(0, board_size * board_size - 1)
        chromosome[idx] = new_pos
    return chromosome


# Genetic algorithm function
def genetic_algorithm(board_size, population_size, generations, mutation_rate):
    population = initialize_population(population_size, board_size)
    best_solutions = []

    for generation in range(generations):
        # Evaluate and sort population by fitness
        population = sorted(population, key=lambda chrom: fitness(chrom, board_size), reverse=True)
        best_solution = population[0]
        best_solutions.append({
            "generation": generation + 1,
            "solution": best_solution,
            "fitness": fitness(best_solution, board_size)
        })

        # Create the next generation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:10], k=2) # select 2 chromosome from top 10
            children = crossover(parent1, parent2)
            child1 = mutate(children[0], mutation_rate, board_size)
            child2 = mutate(children[1], mutation_rate, board_size)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    # Return the best solutions found across all generations
    return best_solutions


# Define the input data model using Pydantic
class BoardInput(BaseModel):
    board_size: int
    generations: int
    population_size: int
    mutation_rate: float


# FastAPI endpoint to get solutions
@app.post("/solutions")
def get_solutions_with_board_size(input: BoardInput):
    board_size = input.board_size
    generations = input.generations
    population_size = input.population_size
    mutation_rate = input.mutation_rate

    # Run the genetic algorithm
    solutions = genetic_algorithm(board_size, population_size, generations, mutation_rate)
    return {"board_size": board_size, "solutions": solutions}
