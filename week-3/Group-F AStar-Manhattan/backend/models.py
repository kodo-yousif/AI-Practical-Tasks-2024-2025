class Node:
    def __init__(self, name, position, heuristic, connections):
        self.name = name
        self.position = position
        self.heuristic = heuristic
        self.connections = connections
        self.g_cost = float('inf')
        self.f_cost = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position,
            "heuristic": self.heuristic,
            "connections": self.connections
        }