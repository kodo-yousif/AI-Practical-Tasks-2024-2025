#python -m uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import heapq
from models import Node

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nodes = {
    "S": Node("S", (0, 0), 10, ["Z", "A", "B"]),
    "Z": Node("Z", (0, 5), 8, ["G", "B", "S"]),
    "B": Node("B", (1, 2), 7, ["Z", "D", "S"]),
    "A": Node("A", (2, 0), 5, ["M", "S"]),
    "M": Node("M", (3, 3), 4, ["G", "A"]),
    "D": Node("D", (4, 2), 6, ["G", "B"]),
    "G": Node("G", (3, 5), 4, ["Z", "M", "D"]),
}

original_heuristics = {
    "S": 10, "Z": 8, "B": 7, "A": 5, 
    "M": 4, "D": 6, "G": 4
}

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star(start_node, goal_node):
    for node in nodes.values():
        node.g_cost = float('inf')
        node.f_cost = float('inf')
        node.parent = None

    start_node.g_cost = 0
    start_node.f_cost = start_node.heuristic
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

            tentative_g_cost = current_node.g_cost + manhattan_distance(
                current_node.position, neighbor.position
            )
            if tentative_g_cost < neighbor.g_cost:
                neighbor.g_cost = tentative_g_cost
                neighbor.f_cost = neighbor.g_cost + neighbor.heuristic
                neighbor.parent = current_node
                heapq.heappush(open_list, (neighbor.f_cost, neighbor))

    return None, None

class PathRequest(BaseModel):
    start_node: str
    goal_node: str

@app.get("/nodes")
async def get_nodes():
    return {name: node.to_dict() for name, node in nodes.items()}

@app.post("/find-path")
async def find_path(request: PathRequest):
    for name, node in nodes.items():
        node.heuristic = original_heuristics[name]
    
    nodes[request.goal_node].heuristic = 0
    
    path, cost = a_star(nodes[request.start_node], nodes[request.goal_node])
    
    return {
        "path": path, 
        "cost": cost,
        "heuristics": {name: node.heuristic for name, node in nodes.items()}
    }

@app.post("/add-node")
async def add_node(node_data: dict):
    name = node_data["name"]
    nodes[name] = Node(
        name=name,
        position=tuple(node_data["position"]),
        heuristic=node_data["heuristic"],
        connections=node_data["connections"]
    )
    return {"message": f"Node {name} added successfully"}

@app.delete("/delete-node/{node_name}")
async def delete_node(node_name: str):
    if node_name in nodes:
        for node in nodes.values():
            if node_name in node.connections:
                node.connections.remove(node_name)
        
        del nodes[node_name]
        return {"message": f"Node {node_name} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Node not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 