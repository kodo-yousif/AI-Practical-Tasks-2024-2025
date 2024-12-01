import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import colorsys
from datetime import datetime

class City:
    cityCount = 0
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name or f"City {City.cityCount}"
        City.cityCount += 1


class TSPGeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.01, elite_size=5, convergence_threshold=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.convergence_threshold = convergence_threshold
        self.cities = []
        self.population = []
        self.generation_history = []
        self.best_fitness_count = 0
        self.last_best_fitness = 0
        self.best_solution = None
        self.best_distance = float('inf')

    # euclidean distance
    def calculate_distance(self, city1, city2):
        return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

    def calculate_fitness(self, route):
        total_distance = 0
        for i in range(len(route)):
            from_city = self.cities[route[i]]
            to_city = self.cities[route[(i + 1) % len(route)]]
            total_distance += self.calculate_distance(from_city, to_city)
        return 1 / total_distance

    def create_initial_population(self):
        self.population = []
        for _ in range(self.population_size):
            route = list(range(len(self.cities)))
            random.shuffle(route)
            self.population.append(route)
        self.best_fitness_count = 0
        self.last_best_fitness = 0
        self.generation_history = []
        self.best_solution = None
        self.best_distance = float('inf')


    def select_parent(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        for i, fitness in enumerate(fitness_scores):
            current_sum += fitness
            if current_sum >= selection_point:
                return self.population[i]
        return self.population[-1]
    


    # partially mapped crossover (PMX)
    def crossover(self, parent1, parent2):
        child = [-1] * len(parent1)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child[start:end + 1] = parent1[start:end + 1]
        parent2_idx = 0
        child_idx = 0
        while -1 in child:

            if parent2[parent2_idx] not in child:
                while child_idx < len(child) and child[child_idx] != -1:
                    child_idx += 1
                
                if child_idx < len(child):
                    child[child_idx] = parent2[parent2_idx]
                
            parent2_idx += 1
        return child

    def mutate(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def check_convergence(self, current_best_fitness):
        if abs(current_best_fitness - self.last_best_fitness) < 1e-6:
            self.best_fitness_count += 1
        else:
            self.best_fitness_count = 0
            self.last_best_fitness = current_best_fitness
        
        return self.best_fitness_count >= self.convergence_threshold

    def get_total_distance(self, route):
        total_distance = 0
        for i in range(len(route)):
            from_city = self.cities[route[i]]
            to_city = self.cities[route[(i + 1) % len(route)]]
            total_distance += self.calculate_distance(from_city, to_city)
        return total_distance

    def evolve(self):
        fitness_scores = [self.calculate_fitness(route) for route in self.population]
        combined = list(zip(fitness_scores, self.population))
        combined.sort(reverse=True)
        fitness_scores, self.population = zip(*combined)
        self.population = list(self.population)
        
        best_fitness = fitness_scores[0]
        best_route = self.population[0].copy()
        current_distance = self.get_total_distance(best_route)
        
        if current_distance < self.best_distance:
            self.best_distance = current_distance
            self.best_solution = best_route.copy()
            
        self.generation_history.append((best_route, best_fitness))
        
        new_population = self.population[:self.elite_size]
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness_scores)
            parent2 = self.select_parent(fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        has_converged = self.check_convergence(best_fitness)
        return best_route, best_fitness, has_converged

class TSPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Genetic Algorithm Solver")
        self.root.geometry("1200x800")
        
        self.style = ttk.Style()
        self.style.configure("TButton", padding=5)
        self.style.configure("TLabel", padding=3)
        
        self.ga = TSPGeneticAlgorithm()
        self.setup_variables()
        self.setup_gui()
        
    def setup_variables(self):
        self.current_generation = 0
        self.max_generations = tk.IntVar(value=1000)
        self.population_size = tk.IntVar(value=50)
        self.mutation_rate = tk.DoubleVar(value=0.01)
        self.elite_size = tk.IntVar(value=5)
        self.convergence_threshold = tk.IntVar(value=100)
        self.num_cities = tk.IntVar(value=10)
        self.best_distance = tk.DoubleVar(value=0)
        self.animation_speed = tk.IntVar(value=100)
        self.is_evolving = False

    def setup_gui(self):
        
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(left_panel, text="Algorithm Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        param_entries = [
            ("Number of Cities:", self.num_cities),
            ("Population Size:", self.population_size),
            ("Mutation Rate:", self.mutation_rate),
            ("Elite Size:", self.elite_size),
            ("Max Generations:", self.max_generations),
            ("Convergence Threshold:", self.convergence_threshold),
            ("Animation Speed (ms):", self.animation_speed)
        ]
        
        for i, (label, var) in enumerate(param_entries):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(params_frame, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=2)
        
        # Control buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Add Random Cities", command=self.add_random_cities).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Clear Cities", command=self.clear_cities).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Start Evolution", command=self.start_evolution).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Stop Evolution", command=self.stop_evolution).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=2)
        
        # Status and Statistics Frame
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=5, width=30)
        self.stats_text.pack(fill=tk.X)
        
        # History Frame
        history_frame = ttk.LabelFrame(left_panel, text="Generation History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        history_scroll = ttk.Scrollbar(history_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(history_frame, yscrollcommand=history_scroll.set)
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_select_generation)
        history_scroll.config(command=self.history_listbox.yview)
        
        # Right panel with canvas
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)
        
        self.canvas = tk.Canvas(right_panel, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.on_canvas_resize)

    def on_canvas_resize(self, event):
        if self.ga.cities:
            self.draw_cities()

    def update_stats(self, generation, distance):
        stats = (
            f"Generation: {generation}\n"
            f"Current Distance: {distance:.2f}\n"
            f"Best Distance: {self.best_distance.get():.2f}\n"
            f"Number of Cities: {len(self.ga.cities)}\n"
            f"Population Size: {self.ga.population_size}"
        )
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)

    def get_color(self, i, total):
        hue = i / total
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

    def draw_cities(self, route=None):
        self.canvas.delete("all")
        
        # Draw cities
        for i, city in enumerate(self.ga.cities):
            self.canvas.create_oval(city.x-5, city.y-5, city.x+5, city.y+5, fill="red")
            self.canvas.create_text(city.x, city.y-15, text=city.name)
            
        # Draw route if provided
        if route:
            for i in range(len(route)):
                city1 = self.ga.cities[route[i]]
                city2 = self.ga.cities[route[(i + 1) % len(route)]]
                color = self.get_color(i, len(route))
                self.canvas.create_line(city1.x, city1.y, city2.x, city2.y, fill=color)

    def add_random_cities(self):
        self.clear_cities()
        num_cities = self.num_cities.get()
        City.cityCount = 0
        for i in range(num_cities):
            x = random.randint(50, 600)
            y = random.randint(50, 600)
            self.ga.cities.append(City(x, y))
            
        # Update GA parameters
        self.ga.population_size = self.population_size.get()
        self.ga.mutation_rate = self.mutation_rate.get()
        self.ga.elite_size = self.elite_size.get()
        self.ga.convergence_threshold = self.convergence_threshold.get()
        
        self.ga.create_initial_population()
        self.draw_cities()
        self.update_stats(0, float('inf'))

    def clear_cities(self):
        self.ga.cities = []
        self.ga.population = []
        self.ga.generation_history = []
        self.current_generation = 0
        self.best_distance.set(0)
        self.canvas.delete("all")
        self.history_listbox.delete(0, tk.END)
        self.stats_text.delete(1.0, tk.END)

    def start_evolution(self):
        if not self.ga.cities:
            messagebox.showwarning("Warning", "Please add cities first!")
            return
        self.is_evolving = True
        self.evolve_step()

    def stop_evolution(self):
        self.is_evolving = False

    def evolve_step(self):
        if not self.is_evolving:
            return
            
        best_route, best_fitness, has_converged = self.ga.evolve()
        current_distance = self.ga.get_total_distance(best_route)
         
        self.current_generation += 1
        self.best_distance.set(self.ga.best_distance)
        
        # Update display
        self.draw_cities(best_route)
        self.update_stats(self.current_generation, current_distance)
        self.history_listbox.insert(tk.END, f"Gen {self.current_generation}: {current_distance:.2f}")
        self.history_listbox.see(tk.END)
        
        # Check stopping conditions
        if self.current_generation >= self.max_generations.get():
            self.is_evolving = False
            self.draw_cities(self.ga.best_solution)  # Display best solution
            messagebox.showinfo("Complete", f"Evolution completed!\nBest distance: {self.ga.best_distance:.2f}")
        elif has_converged:
            self.is_evolving = False
            self.draw_cities(self.ga.best_solution)  # Display best solution
            messagebox.showinfo("Converged", f"Solution converged!\nBest distance: {self.ga.best_distance:.2f}")
        
        if self.is_evolving:
            self.root.after(self.animation_speed.get(), self.evolve_step)

    def on_select_generation(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            route, _ = self.ga.generation_history[index]
            self.draw_cities(route)

    def save_results(self):
        if not self.ga.generation_history:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tsp_results_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write("TSP Genetic Algorithm Results\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Number of Cities: {len(self.ga.cities)}\n")
                f.write(f"Best Distance: {self.best_distance.get():.2f}\n")
                
                # Get the city names from the best solution
                best_route_names = [self.ga.cities[city_index].name for city_index in self.ga.best_solution]
                f.write(f"Best Route: {', '.join(best_route_names)}\n\n")
                
                f.write("Generation History:\n")
                
                for i, (route, fitness) in enumerate(self.ga.generation_history):
                    distance = 1 / fitness  # Invert fitness to get distance
                    f.write(f"Generation {i+1}: Distance = {distance:.2f}\n")
                    
            messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGUI(root)
    root.mainloop()