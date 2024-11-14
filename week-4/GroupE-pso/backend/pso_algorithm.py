import random
import math

class Particle:
    def __init__(self, goal_x, goal_y):
        self.position = [random.uniform(-10, 10), random.uniform(-10, 10)]
        self.velocity = [0, 0]
        self.best_position = self.position[:]
        self.best_value = float("inf")
        self.goal = [goal_x, goal_y]

    def update_velocity(self, global_best, cognitive_coeff, social_coeff, inertia):
        r1 = random.random()
        r2 = random.random()
        for i in range(2):
            cognitive = cognitive_coeff * r1 * (self.best_position[i] - self.position[i])
            social = social_coeff * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive + social

    def update_position(self):
        for i in range(2):
            self.position[i] += self.velocity[i]
    
    def evaluate_fitness(self):
        distance = math.sqrt((self.position[0] - self.goal[0])**2 + (self.position[1] - self.goal[1])**2)
        if distance < self.best_value:
            self.best_value = distance
            self.best_position = self.position[:]
        return distance

def run_pso(goal_x, goal_y, num_particles, cognitive_coeff, social_coeff, inertia, iterations=100):
    particles = [Particle(goal_x, goal_y) for _ in range(num_particles)]
    global_best_position = [random.uniform(-10, 10), random.uniform(-10, 10)]
    global_best_value = float("inf")
    generation_data = []

    for generation in range(iterations):
        for particle in particles:
            fitness = particle.evaluate_fitness()
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = particle.position[:]

        generation_data.append({
            "generation": generation,
            "global_best_position": global_best_position,
            "global_best_value": global_best_value,
            "particles": [(p.position, p.velocity) for p in particles]
        })

        for particle in particles:
            particle.update_velocity(global_best_position, cognitive_coeff, social_coeff, inertia)
            particle.update_position()
    
    return generation_data
