from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Node:
    def __init__(self, node_id, name, value):
        self.id = node_id
        self.name = name
        self.value = value
        self.x = value % 10  # Simple way to generate x coordinate
        self.y = value // 10  # Simple way to generate y coordinate
        self.neighbors = []
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None

    def add_neighbor(self, neighbor_node, cost):
        self.neighbors.append((neighbor_node, cost))

    def __lt__(self, other):
        return self.f < other.f



def euclidean_distance(node1, node2):
    return ((node2.x - node1.x)**2 + (node2.y - node1.y)**2) ** 0.5

def build_graph(nodes_info, edges_info):
    graph = {}
    
    for node_info in nodes_info:
        node_id = node_info["id"]
        name = node_info["name"]
        value = node_info["value"]
        graph[node_id] = Node(node_id, name, value)

    for edge_info in edges_info:
        source_id = edge_info["source"]
        target_id = edge_info["target"]
        cost = edge_info["cost"]
        graph[source_id].add_neighbor(graph[target_id], cost)
        graph[target_id].add_neighbor(graph[source_id], cost)

    return graph

def reconstruct_path(current_node):
    path = []
    path_nodes = []
    total_cost = current_node.g
    
    while current_node:
        path.append((current_node.id, current_node.name))
        path_nodes.append(current_node.id)
        current_node = current_node.parent
    
    return path[::-1], path_nodes[::-1], total_cost

def astar(graph, start_id, goal_id):
    open_list = []
    closed_list = set()
    steps = []
    step_counter = 1

    start_node = graph[start_id]
    goal_node = graph[goal_id]

    start_node.g = 0
    start_node.h = euclidean_distance(start_node, goal_node)
    start_node.f = start_node.g + start_node.h
    open_list.append(start_node)

    # Initial step with more detail
    steps.append({
        "step": step_counter,
        "message": f"Process Node {start_node.name}",
        "current": start_node.id,
        "current_name": start_node.name,
        "current_f": start_node.f,
        "current_g": start_node.g,
        "current_h": start_node.h,
        "open_set": [n.id for n in open_list],
        "open_set_details": [(n.name, n.f) for n in open_list],
        "closed_set": list(closed_list),
        "neighbors_info": [],
        "f_value": start_node.f
    })
    step_counter += 1

    while open_list:
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.add(current_node.id)

        # Gather neighbor information
        neighbors_info = []
        for neighbor, cost in current_node.neighbors:
            if neighbor.id not in closed_list:
                tentative_g = current_node.g + cost
                old_g = neighbor.g
                neighbors_info.append({
                    "name": neighbor.name,
                    "g": tentative_g,
                    "h": euclidean_distance(neighbor, goal_node),
                    "f": tentative_g + euclidean_distance(neighbor, goal_node),
                    "better_path": tentative_g < old_g if old_g != float('inf') else None
                })

        steps.append({
            "step": step_counter,
            "message": f"Process Node {current_node.name}",
            "current": current_node.id,
            "current_name": current_node.name,
            "current_f": current_node.f,
            "current_g": current_node.g,
            "current_h": current_node.h,
            "open_set": [n.id for n in open_list],
            "open_set_details": [(n.name, n.f) for n in sorted(open_list, key=lambda x: x.f)],
            "closed_set": list(closed_list),
            "neighbors_info": neighbors_info,
            "f_value": current_node.f
        })
        step_counter += 1

        if current_node.id == goal_id:
            path, path_nodes, total_cost = reconstruct_path(current_node)
            
            # Add path information to the final step
            final_step = {
                "step": step_counter,
                "message": f"Goal node '{current_node.name}' reached.",
                "current": current_node.id,
                "current_name": current_node.name,
                "current_f": current_node.f,
                "current_g": current_node.g,
                "current_h": current_node.h,
                "open_set": [n.id for n in open_list],
                "open_set_details": [(n.name, n.f) for n in sorted(open_list, key=lambda x: x.f)],
                "closed_set": list(closed_list),
                "neighbors_info": neighbors_info,
                "f_value": current_node.f,
                "is_goal": True,
                "path": path,  # Add the path
                "total_cost": total_cost  # Add the total cost
            }
            steps.append(final_step)  # Add the final step to steps
            
            return {
                "path": path,
                "path_nodes": path_nodes,
                "total_cost": total_cost,
                "steps": steps
            }

        for neighbor, cost in current_node.neighbors:
            if neighbor.id in closed_list:
                continue

            tentative_g = current_node.g + cost

            if tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = euclidean_distance(neighbor, goal_node)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_list:
                    open_list.append(neighbor)

    return None

@app.post("/find-path")
async def find_path(data: Dict[str, Any]):
    try:
        graph = build_graph(data["nodes"], data["edges"])
        result = astar(graph, data["startNode"], data["endNode"])
        
        if result is None:
            raise HTTPException(status_code=404, detail="No path found")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
