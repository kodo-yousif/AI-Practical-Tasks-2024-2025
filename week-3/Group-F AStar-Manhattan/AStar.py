import tkinter as tk
from random import random
from tkinter import messagebox
import heapq

from random import randint

class Node:
    def __init__(self, name, position, connections):
        self.name = name
        self.position = position
        self.connections = connections
        self.g_cost = float('inf')
        self.f_cost = float('inf')
        self.parent = None

    def get_heuristic(self, goal_position):
        # Return zero if the node is the goal node; otherwise, return a random heuristic value
        if self.position == goal_position:
            return 0
        return randint(1, 10)

    def __lt__(self, other):
        return self.f_cost < other.f_cost



def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

nodes = {
    "S": Node("S", (0, 0),  ["Z", "A", "B"]),
    "Z": Node("Z", (0, 5),  ["G", "B", "S"]),
    "B": Node("B", (1, 2),  ["Z", "D", "S"]),
    "A": Node("A", (2, 0),  ["M", "S"]),
    "M": Node("M", (3, 3),  ["G", "A"]),
    "D": Node("D", (4, 2),  ["G", "B"]),
    "G": Node("G", (5, 5),  ["Z", "M", "D"]),
}

def a_star(start_node, goal_node):
    start_node.g_cost = 0
    start_node.f_cost = start_node.get_heuristic(goal_node.position)  # Use dynamic heuristic
    open_list = []
    heapq.heappush(open_list, (start_node.f_cost, start_node))
    closed_set = set()

    while open_list:
        _, current_node = heapq.heappop(open_list)
        closed_set.add(current_node.name)

        if current_node.name == goal_node.name:
            path = []
            node = current_node
            while node:
                path.append(node.name)
                node = node.parent
            return path[::-1], current_node.g_cost

        for neighbor_name in current_node.connections:
            neighbor = nodes[neighbor_name]
            if neighbor.name in closed_set:
                continue

            tentative_g_cost = current_node.g_cost + manhattan_distance(current_node.position, neighbor.position)
            if tentative_g_cost < neighbor.g_cost:
                neighbor.g_cost = tentative_g_cost
                neighbor.f_cost = neighbor.g_cost + neighbor.get_heuristic(goal_node.position)
                neighbor.parent = current_node
                heapq.heappush(open_list, (neighbor.f_cost, neighbor))

    return None, None

class AStarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("A* Pathfinding with Manhattan Distance")

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.grid(row=0, column=0)

        self.controls_frame = tk.Frame(root)
        self.controls_frame.grid(row=1, column=0, pady=5)

        self.start_label = tk.Label(self.controls_frame, text="Start Node:")
        self.start_label.grid(row=0, column=0, padx=10, sticky="w")
        self.goal_label = tk.Label(self.controls_frame, text="Goal Node:")
        self.goal_label.grid(row=0, column=1, padx=10, sticky="w")

        self.start_node_var = tk.StringVar(root)
        self.goal_node_var = tk.StringVar(root)
        self.start_node_menu = tk.OptionMenu(self.controls_frame, self.start_node_var, *nodes.keys())
        self.goal_node_menu = tk.OptionMenu(self.controls_frame, self.goal_node_var, *nodes.keys())
        self.start_node_menu.grid(row=1, column=0, padx=10)
        self.goal_node_menu.grid(row=1, column=1, padx=10)

        self.find_path_button = tk.Button(self.controls_frame, text="Find Path", command=self.show_path)
        self.find_path_button.grid(row=2, column=0, columnspan=2, pady=5)
        self.reset_button = tk.Button(self.controls_frame, text="Reset", command=self.reset_nodes)
        self.reset_button.grid(row=3, column=0, columnspan=2, pady=5)

        self.draw_grid()
        self.draw_nodes()
        self.draw_connections()

    def draw_grid(self):
        for i in range(6):
            for j in range(6):
                self.canvas.create_rectangle(50 * j, 50 * i, 50 * (j + 1), 50 * (i + 1), outline="gray")

    def draw_connections(self):
        for node in nodes.values():
            x1, y1 = node.position[1] * 50 + 25, node.position[0] * 50 + 25
            for neighbor_name in node.connections:
                neighbor = nodes[neighbor_name]
                x2, y2 = neighbor.position[1] * 50 + 25, neighbor.position[0] * 50 + 25
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                distance = manhattan_distance(node.position, neighbor.position)
                self.canvas.create_text(mid_x, mid_y, text=str(distance), fill="blue", font=("Arial", 12))

    def draw_nodes(self):
        goal_position = nodes[self.goal_node_var.get()].position if self.goal_node_var.get() in nodes else (0, 0)
        for node in nodes.values():
            x, y = node.position[1] * 50 + 25, node.position[0] * 50 + 25
            self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="#F8EDE3", outline="black", width=1)
            self.canvas.create_text(x, y - 5, text=node.name, font=("Arial", 8, "bold"), fill="black")
            heuristic = node.get_heuristic(goal_position)
            self.canvas.create_text(x, y + 10, text=f"H:{heuristic}", font=("Arial", 8), fill="black")

    def show_path(self):
        start_node_name = self.start_node_var.get()
        goal_node_name = self.goal_node_var.get()

        if not start_node_name or not goal_node_name:
            messagebox.showerror("Error", "Please select both start and goal nodes.")
            return

        self.reset_nodes_data()
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_nodes()
        self.draw_connections()

        path, total_cost = a_star(nodes[start_node_name], nodes[goal_node_name])
        if path:
            for node_name in path:
                node = nodes[node_name]
                x, y = node.position[1] * 50 + 25, node.position[0] * 50 + 25
                outline_color = "orange" if node_name not in {start_node_name, goal_node_name} else "green" if node_name == start_node_name else "red"
                self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, outline=outline_color, width=3)
            messagebox.showinfo("Path Found", f"Path found with total cost: {total_cost}")
        else:
            messagebox.showinfo("No Path", "No path found.")

    def reset_nodes_data(self):
        for node in nodes.values():
            node.g_cost = float('inf')
            node.f_cost = float('inf')
            node.parent = None

    def reset_nodes(self):
        self.start_node_var.set(None)
        self.goal_node_var.set(None)
        self.reset_nodes_data()
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_nodes()
        self.draw_connections()

root = tk.Tk()
app = AStarGUI(root)
root.mainloop()