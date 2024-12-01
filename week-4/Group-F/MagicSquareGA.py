import random
import tkinter as tk
from tkinter import ttk, messagebox

class MagicSquareGA:
    def __init__(self, population_size=30, generations=1000, mutation_rate=0.01):
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
        child = [None]*size

        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        fill_values = [item for item in parent2 if item not in child]
        pointer = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill_values[pointer]
                pointer +=1

        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def evolve(self):
        self.create_population()

        for generation in range(self.generations):
            # Evaluate fitness and find the best individual
            fitness_scores = [(self.fitness(individual), individual) for individual in self.population]
            fitness_scores.sort(key=lambda x: x[0])
            best_fitness, best_individual = fitness_scores[0]
            self.best_solutions.append({'generation': generation, 'fitness': best_fitness, 'individual': best_individual})

            if best_fitness == 0:
                print(f"Magic Square found at generation {generation}")
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


class MagicSquareGUI:
    def __init__(self, master, ga):
        self.master = master
        self.ga = ga
        self.master.title("Magic Square Puzzle")
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.run_button = ttk.Button(main_frame, text="Run Genetic Algorithm", command=self.run_ga)
        self.run_button.grid(row=0, column=0, columnspan=2, pady=10)

        self.reset_button = ttk.Button(main_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.listbox = tk.Listbox(main_frame, width=40, height=15)
        self.listbox.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        self.grid_frame = ttk.LabelFrame(main_frame, text="Magic Square", padding="10 10 10 10")
        self.grid_frame.grid(row=0, column=2, rowspan=2, padx=20, sticky=(tk.N))

        self.grid_labels = []
        for row in range(3):
            row_labels = []
            for col in range(3):
                label = ttk.Label(self.grid_frame, text="", borderwidth=2, relief="ridge", width=5, anchor="center",
                                  background="white", font=("Helvetica", 18))
                label.grid(row=row, column=col, padx=5, pady=5)
                row_labels.append(label)
            self.grid_labels.append(row_labels)

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def run_ga(self):
        self.run_button.config(state=tk.DISABLED)
        self.ga.evolve()
        self.listbox.delete(0, tk.END)

        found_solution = False

        for solution in self.ga.best_solutions:
            self.listbox.insert(tk.END, f"Generation {solution['generation']}, Fitness: {solution['fitness']}")
            if solution['fitness'] == 0 and not found_solution:
                self.display_solution(solution['individual'])
                found_solution = True

        if not found_solution:
            # Clear the grid if no perfect solution is found
            for row_labels in self.grid_labels:
                for label in row_labels:
                    label['text'] = ""

        self.run_button.config(state=tk.NORMAL)
        messagebox.showinfo("Info", "Genetic Algorithm has completed running.")

    def reset(self):
        self.ga.population = []
        self.ga.best_solutions = []

        self.listbox.delete(0, tk.END)

        for row_labels in self.grid_labels:
            for label in row_labels:
                label['text'] = ""

        self.run_button.config(state=tk.NORMAL)

        messagebox.showinfo("Info", "The Genetic Algorithm has been reset.")

    def on_select(self, event):
        if not self.listbox.curselection():
            return
        index = self.listbox.curselection()[0]
        solution = self.ga.best_solutions[index]['individual']
        self.display_solution(solution)

    def display_solution(self, solution):
        for i in range(9):
            row = i // 3
            col = i % 3
            self.grid_labels[row][col]['text'] = solution[i]


if __name__ == "__main__":
    ga = MagicSquareGA()
    root = tk.Tk()
    app = MagicSquareGUI(root, ga)
    root.mainloop()
