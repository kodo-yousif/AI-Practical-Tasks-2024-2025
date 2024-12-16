import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Learning Algorithm Visualization")
        
        # Initialize parameters
        self.learning_rate = 0.03
        self.weights = np.random.randn(3)  # [w1, w2, bias]
        print(self.weights)
        self.fps = 2
        self.max_iterations = 100  # Default max iterations
        
        # Initialize empty data lists
        self.class1 = []
        self.class2 = []
        self.current_class = 1  # For clicking mode
        
        self.setup_ui()
        self.setup_plot()
        
        # Create log file (text format)
        self.log_file = open('perceptron_iterations.txt', 'w')
        self.log_file.write("Iteration  | w1       | w2       | bias     | Misclassified | Error Rate\n")
        
        self.iteration = 0
        self.is_running = False
        self.click_mode = False

    def setup_ui(self):
        # Main Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Data Generation Frame
        data_frame = ttk.LabelFrame(control_frame, text="Data Generation")
        data_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Number of points input
        ttk.Label(data_frame, text="Points per class:").pack(side=tk.LEFT)
        self.points_entry = ttk.Entry(data_frame, width=10)
        self.points_entry.insert(0, "10")
        self.points_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_frame, text="Generate Random Data", command=self.generate_random_data).pack(side=tk.LEFT, padx=5)
        
        # Click mode toggle
        self.click_button = ttk.Button(data_frame, text="Enable Click Mode", command=self.toggle_click_mode)
        self.click_button.pack(side=tk.LEFT, padx=5)
        
        self.class_label = ttk.Label(data_frame, text="Current Class: 1")
        self.class_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_frame, text="Switch Class", command=self.switch_class).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_frame, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        
        # Algorithm Control Frame
        algo_frame = ttk.LabelFrame(control_frame, text="Algorithm Control")
        algo_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Learning Rate Input
        ttk.Label(algo_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_entry = ttk.Entry(algo_frame, width=10)
        self.lr_entry.insert(0, str(self.learning_rate))
        self.lr_entry.pack(side=tk.LEFT, padx=5)
        
        # FPS Input
        ttk.Label(algo_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_entry = ttk.Entry(algo_frame, width=10)
        self.fps_entry.insert(0, str(self.fps))
        self.fps_entry.pack(side=tk.LEFT, padx=5)
        
        # Max Iterations Input
        ttk.Label(algo_frame, text="Max Iterations:").pack(side=tk.LEFT)
        self.max_iter_entry = ttk.Entry(algo_frame, width=10)
        self.max_iter_entry.insert(0, str(self.max_iterations))
        self.max_iter_entry.pack(side=tk.LEFT, padx=5)
        
        # Iteration Counter Display
        self.iter_label = ttk.Label(algo_frame, text="Iteration: 0")
        self.iter_label.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        self.start_button = ttk.Button(algo_frame, text="Start", command=self.toggle_animation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(algo_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)

    def setup_plot(self):
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        
        # Setup canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        # Bind click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.update_plot()

    def generate_random_data(self):
        try:
            n_points = int(self.points_entry.get())
            if n_points <= 0:
                raise ValueError("Number of points must be positive")
                
            self.class1 = (np.random.randn(n_points, 2) + np.array([2, 2])).tolist()
            self.class2 = (np.random.randn(n_points, 2) - np.array([2, 2])).tolist()
            self.update_plot()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def on_click(self, event):
        if not self.click_mode or event.inaxes != self.ax:
            return
        
        if self.current_class == 1:
            self.class1.append([event.xdata, event.ydata])
        else:
            self.class2.append([event.xdata, event.ydata])
        
        self.update_plot()

    def toggle_click_mode(self):
        self.click_mode = not self.click_mode
        self.click_button.config(text="Disable Click Mode" if self.click_mode else "Enable Click Mode")

    def switch_class(self):
        self.current_class = 3 - self.current_class  # Switches between 1 and 2
        self.class_label.config(text=f"Current Class: {self.current_class}")

    def clear_data(self):
        self.class1 = []
        self.class2 = []
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        
        # Convert lists to numpy arrays for plotting
        if self.class1:
            class1_array = np.array(self.class1)
            self.ax.scatter(class1_array[:, 0], class1_array[:, 1], 
                          color='blue', label='Class 1')
        
        if self.class2:
            class2_array = np.array(self.class2)
            self.ax.scatter(class2_array[:, 0], class2_array[:, 1], 
                          color='red', label='Class 2')
        
        # Plot decision boundary
        if self.class1 and self.class2:
            x_range = np.array([-6, 6])
            y_range = -(self.weights[0] * x_range + self.weights[2]) / self.weights[1]
            self.ax.plot(x_range, y_range, 'g-', label='Decision Boundary')
        
        self.ax.set_xlim([-6, 6])
        self.ax.set_ylim([-6, 6])
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()

    def toggle_animation(self):
        if len(self.class1) == 0 or len(self.class2) == 0:
            messagebox.showerror("Error", "Please add data points for both classes")
            return
            
        if not self.is_running:
            try:
                self.learning_rate = float(self.lr_entry.get())
                self.fps = float(self.fps_entry.get())
                self.max_iterations = int(self.max_iter_entry.get())
                
                if self.max_iterations <= 0:
                    raise ValueError("Max iterations must be positive")
                if self.fps <= 0:
                    raise ValueError("FPS must be positive")
                if self.learning_rate <= 0:
                    raise ValueError("Learning rate must be positive")
                
                self.is_running = True
                self.start_button.config(text="Stop")
                self.animate()
            except ValueError as e:
                messagebox.showerror("Error", str(e))
        else:
            self.is_running = False
            self.start_button.config(text="Start")

    def reset(self):
        self.is_running = False
        self.start_button.config(text="Start")
        self.weights = np.random.randn(3)
        self.iteration = 0
        self.iter_label.config(text=f"Iteration: {self.iteration}")
        self.update_plot()

    def train_step(self):
        misclassified = 0 
        X = np.vstack((self.class1, self.class2)) #[c1,c1,c1,c1,c2,c2,c2,c2,c2] c1 = (x,y)
        y = np.array([1]*len(self.class1) + [-1]*len(self.class2)) #[1,1,1,1,-1,-1,-1,-1,-1]
        
        for i in range(len(X)):
            x = np.append(X[i], 1)  # Add bias term | [x1,x2, 1] | x1 = x inside class | x2 = y inside class
            prediction = np.sign(np.dot(self.weights, x)) 
            
            if prediction != y[i]: #prediction = x[i] after dot product with weights
                misclassified += 1
                self.weights += self.learning_rate * y[i] * x
        
        error_rate = misclassified / len(X) * 100  # Calculate error rate
        # Log iteration data to text file
        self.log_file.write(f"{self.iteration:<10} | {self.weights[0]:<8.4f} | {self.weights[1]:<8.4f} | "
                            f"{self.weights[2]:<8.4f} | {misclassified:<13} | {error_rate:<10.2f}%\n")
        self.log_file.flush()  # Ensure data is written immediately
        return misclassified

    def animate(self):
        if not self.is_running:
            return
            
        misclassified = self.train_step()
        self.iteration += 1
        self.iter_label.config(text=f"Iteration: {self.iteration}")
        self.update_plot()
        
        # Check both convergence and iteration limit
        if misclassified > 0 and self.iteration < self.max_iterations:
            self.root.after(int(1000/self.fps), self.animate)
        else:
            self.is_running = False
            self.start_button.config(text="Start")
            if misclassified == 0:
                messagebox.showinfo("Success", "Convergence achieved!")
            else:
                messagebox.showinfo("Stopped", f"Maximum iterations ({self.max_iterations}) reached")

def main():
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
    app.log_file.close()  # Close the log file when the window is closed

if __name__ == "__main__":
    main()
