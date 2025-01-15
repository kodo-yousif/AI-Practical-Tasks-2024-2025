# run code command : [python magic_square.py]


import random
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class MagicSquareGA:
    def __init__(self, population_size=200, generations=1000, mutation_rate=0.5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.magic_constant = 15
        self.numbers = list(range(1, 10))
        self.population = []
        self.best_solutions = []

    def create_individual(self):
        individual = self.numbers[:]
        random.shuffle(individual)
        return individual

    def create_population(self):
        self.population = [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        rows = [individual[0:3], individual[3:6], individual[6:9]]
        cols = [individual[0:9:3], individual[1:9:3], individual[2:9:3]]
        diags = [individual[0:9:4], individual[2:7:2]]

        total_diff = 0
        for line in rows + cols + diags:
            total_diff += abs(sum(line) - self.magic_constant)
        return total_diff

    def selection(self):
        tournament_size = 5
        selected = []
        for _ in range(self.population_size):
            contestants = random.sample(self.population, tournament_size)
            contestants.sort(key=lambda x: self.fitness(x))
            selected.append(contestants[0])
        return selected

    def crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        crossover_point = random.randint(0, size - 1)
        child[:crossover_point] = parent1[:crossover_point]
        remaining_values = [item for item in parent2 if item not in child]
        child[crossover_point:] = remaining_values
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evolve(self):
        self.create_population()
        
        for generation in range(self.generations):
            fitness_scores = [(self.fitness(individual), individual) for individual in self.population]
            fitness_scores.sort(key=lambda x: x[0])
            best_fitness, best_individual = fitness_scores[0]
            
            self.best_solutions.append({
                'generation': generation,
                'fitness': best_fitness,
                'individual': best_individual
            })

            if best_fitness == 0:
                break

            selected = self.selection()
            next_generation = []
            
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.population_size else selected[0]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_generation.extend([child1, child2])

            self.population = next_generation[:self.population_size]

ga = MagicSquareGA()

@app.route('/generate', methods=['GET'])
def generate_magic_square():
    ga.evolve()
    solutions = []
    for solution in ga.best_solutions:
        solutions.append({
            'generation': solution['generation'],
            'fitness': solution['fitness'],
            'individual': solution['individual']
        })
    return jsonify({
        'solutions': solutions,
        'found': any(s['fitness'] == 0 for s in solutions)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 