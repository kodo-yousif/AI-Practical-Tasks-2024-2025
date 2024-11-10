import tkinter as tk
from tkinter import ttk, messagebox
import math
from typing import Dict, List, Set, Tuple
import heapq

class Node:
    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.x = x
        self.y = y
        self.connections = {}  # name -> cost
        self.parent = None
        self.g_cost = float('inf')
        self.h_cost = float('inf')  # Manual heuristic cost
        self.f_cost = float('inf')

class PathFinder:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        
    def chebyshev_distance(self, node1: Node, node2: Node) -> float:
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        return max(dx, dy)
    
    def a_star(self, start_name: str, goal_name: str) -> List[str]:
        if start_name not in self.nodes or goal_name not in self.nodes:
            return []
        
        for node in self.nodes.values():
            node.parent = None
            node.g_cost = float('inf')
            node.f_cost = float('inf')
        
        start_node = self.nodes[start_name]
        start_node.g_cost = 0
        start_node.f_cost = start_node.h_cost  # Using manual heuristic
        
        open_set = [(start_node.f_cost, start_name)]
        closed_set: Set[str] = set()

        print(f"Starting A* from {start_name} to {goal_name}")
        
        while open_set:
            current_name = heapq.heappop(open_set)[1]
            current = self.nodes[current_name]
            

            print(f"\nCurrent Node: {current_name} (g_cost: {current.g_cost}, f_cost: {current.f_cost})")



            if current_name == goal_name:
                path = []
                while current:
                    path.append(current.name)
                    current = current.parent
                    print(f"Goal {goal_name} reached! Path: {path[::-1]}")
                return path[::-1]
            
            closed_set.add(current_name)
            
            for neighbor_name, cost in current.connections.items():
                if neighbor_name in closed_set:
                    print(f"  Skipping neighbor {neighbor_name} (already in closed set)")
                    continue
                    
                neighbor = self.nodes[neighbor_name]
                tentative_g = current.g_cost + cost
                
                print(f"  Checking neighbor: {neighbor_name} (current neighbor g_cost: {neighbor.g_cost}, tentative_g: {tentative_g})")


                if tentative_g < neighbor.g_cost:
                    print(f"    Found a shorter path to {neighbor_name}! Updating costs and parent.")
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    heapq.heappush(open_set, (neighbor.f_cost, neighbor_name))
                    print(f"    Added {neighbor_name} to open set with f_cost: {neighbor.f_cost}")
                else:
                    print(f"    No update for {neighbor_name}, current path is shorter.")        
        return []

class PathFinderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("A* Pathfinder")
        self.path_finder = PathFinder()
        self.setup_gui()
        
        # Mouse interaction states
        self.adding_node = False
        self.connecting_nodes = False
        self.first_node = None
        self.node_counter = 1
        
    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Mode selection
        ttk.Label(left_panel, text="Mode Selection").pack()
        self.mode_var = tk.StringVar(value="add")
        ttk.Radiobutton(left_panel, text="Add Nodes", variable=self.mode_var, 
                       value="add").pack()
        ttk.Radiobutton(left_panel, text="Connect Nodes", variable=self.mode_var, 
                       value="connect").pack()
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Heuristic input
        ttk.Label(left_panel, text="Set Node Heuristic").pack()
        self.node_select = ttk.Combobox(left_panel, width=10)
        self.node_select.pack()
        
        ttk.Label(left_panel, text="Heuristic Value:").pack()
        self.heuristic_value = ttk.Entry(left_panel, width=10)
        self.heuristic_value.pack()
        
        ttk.Button(left_panel, text="Set Heuristic", 
                  command=self.set_heuristic).pack(pady=5)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Path finding
        ttk.Label(left_panel, text="Find Path").pack()
        self.start_node = ttk.Combobox(left_panel, width=10)
        self.start_node.pack()
        self.goal_node = ttk.Combobox(left_panel, width=10)
        self.goal_node.pack()
        
        ttk.Button(left_panel, text="Find Path", 
                  command=self.find_path).pack(pady=5)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(left_panel, text="Clear Board", command=self.clear_board).pack(pady=5)
        
        # Canvas for visualization
        self.canvas = tk.Canvas(main_container, width=600, height=600, 
                              bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas bindings
        self.canvas.bind("<Button-1>", self.canvas_clicked)
        self.canvas.bind("<Motion>", self.canvas_motion)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
    def clear_board(self):
        self.path_finder.nodes.clear()
        self.node_counter = 1
        self.first_node = None
        self.update_node_lists()
        self.draw_graph()

    def update_node_lists(self):
        nodes = list(self.path_finder.nodes.keys())
        self.node_select['values'] = nodes
        self.start_node['values'] = nodes
        self.goal_node['values'] = nodes
        
    def canvas_clicked(self, event):
        x, y = event.x, event.y
        grid_x = x // 50
        grid_y = y // 50
        
        if self.mode_var.get() == "add":
            self.add_node_at_position(grid_x, grid_y)
        elif self.mode_var.get() == "connect":
            self.handle_connection_click(x, y)
            
    def add_node_at_position(self, grid_x, grid_y):
        name = f"N{self.node_counter}"
        self.node_counter += 1
        
        # Check if position is already occupied
        for node in self.path_finder.nodes.values():
            if node.x == grid_x and node.y == grid_y:
                messagebox.showerror("Error", "Position already occupied!")
                return
                
        self.path_finder.nodes[name] = Node(name, grid_x, grid_y)
        self.path_finder.nodes[name].h_cost = 0  # Default heuristic
        self.update_node_lists()
        self.draw_graph()
        
    def handle_connection_click(self, x, y):
        clicked_node = self.find_node_at_position(x, y)
        
        if clicked_node:
            if not self.first_node:
                self.first_node = clicked_node
                self.status_var.set(f"Selected {clicked_node}. Click another node to connect.")
            else:
                if self.first_node != clicked_node:
                    self.create_connection(self.first_node, clicked_node)
                self.first_node = None
                self.status_var.set("Ready")
                
    def find_node_at_position(self, x, y) -> str:
        grid_x = x // 50
        grid_y = y // 50
        
        for name, node in self.path_finder.nodes.items():
            if node.x == grid_x and node.y == grid_y:
                return name
        return None
        
    def create_connection(self, node1_name: str, node2_name: str):
        node1 = self.path_finder.nodes[node1_name]
        node2 = self.path_finder.nodes[node2_name]
        
        cost = self.path_finder.chebyshev_distance(node1, node2)
        node1.connections[node2_name] = cost
        node2.connections[node1_name] = cost
        self.draw_graph()
        
    def set_heuristic(self):
        node_name = self.node_select.get()
        try:
            h_value = float(self.heuristic_value.get())
            if node_name in self.path_finder.nodes:
                self.path_finder.nodes[node_name].h_cost = h_value
                messagebox.showinfo("Success", 
                                  f"Heuristic for {node_name} set to {h_value}")
                self.draw_graph()
            else:
                messagebox.showerror("Error", "Please select a valid node!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number!")
            
    def canvas_motion(self, event):
        if self.mode_var.get() == "connect" and self.first_node:
            self.draw_graph()
            x1, y1 = (self.path_finder.nodes[self.first_node].x * 50 + 25,
                     self.path_finder.nodes[self.first_node].y * 50 + 25)
            self.canvas.create_line(x1, y1, event.x, event.y, dash=(4, 4))
            
    def draw_graph(self):
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(0, 600, 50):
            self.canvas.create_line(i, 0, i, 600, fill="gray", dash=(1, 1))
            self.canvas.create_line(0, i, 600, i, fill="gray", dash=(1, 1))
        
        # Draw connections
        for name, node in self.path_finder.nodes.items():
            x1, y1 = node.x * 50 + 25, node.y * 50 + 25
            for conn_name, cost in node.connections.items():
                conn_node = self.path_finder.nodes[conn_name]
                x2, y2 = conn_node.x * 50 + 25, conn_node.y * 50 + 25
                self.canvas.create_line(x1, y1, x2, y2)
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                self.canvas.create_text(mx, my - 10, text=f"{cost:.1f}")
                
        # Draw nodes
        for name, node in self.path_finder.nodes.items():
            x, y = node.x * 50 + 25, node.y * 50 + 25
            self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="white")
            self.canvas.create_text(x, y, text=f"{name}\nh={node.h_cost}")
            
    def find_path(self):
        start = self.start_node.get()
        goal = self.goal_node.get()
        
        if not start or not goal:
            messagebox.showerror("Error", "Please select start and goal nodes!")
            return
            
        path = self.path_finder.a_star(start, goal)
        if path:
            self.show_solution(path)
        else:
            messagebox.showerror("Error", "No path found!")
            
    def show_solution(self, path: List[str]):
        solution_window = tk.Toplevel(self.root)
        solution_window.title("Solution Path")
        
        text = tk.Text(solution_window, height=10, width=40)
        text.pack(padx=10, pady=10)
        
        # Calculate total path cost
        total_cost = 0
        for i in range(len(path) - 1):
            node = self.path_finder.nodes[path[i]]
            next_node = self.path_finder.nodes[path[i + 1]]
            total_cost += node.connections[path[i + 1]]
        
        text.insert(tk.END, f"Path found (Total cost: {total_cost:.1f}):\n")
        text.insert(tk.END, " -> ".join(path))
        
        # Highlight path on main canvas
        for i in range(len(path) - 1):
            node1 = self.path_finder.nodes[path[i]]
            node2 = self.path_finder.nodes[path[i + 1]]
            x1, y1 = node1.x * 50 + 25, node1.y * 50 + 25
            x2, y2 = node2.x * 50 + 25, node2.y * 50 + 25
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
            
    def run(self):
        self.root.mainloop()

app = PathFinderGUI()
app.run()