import random
import math
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
# Pydantic is used for request validation.

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# BaseModel It ensures that num_particles and iterations are greater than 0.
# All required fields are provided and of the correct type.
class PSORequest(BaseModel):
    num_particles: int = Field(gt=0, description="Number of particles must be greater than 0")
    goal_x: float
    goal_y: float
    cognitive_coeff: float
    social_coeff: float
    inertia: float
    iterations: int = Field(gt=0, description="Iterations must be greater than 0")


def calculate_fitness(position: Dict[str, float], goal_x: float, goal_y: float) -> float:
    # Calculate Euclidean distance as fitness metric.
    return ((position["x"] - goal_x) ** 2 + 
            (position["y"] - goal_y) ** 2) ** 0.5
    
    
def calculate_pso(params: Dict[str, Any]) -> Dict[str, Any]:
    num_particles = params["num_particles"]
    goal_x = params["goal_x"]
    goal_y = params["goal_y"]
    cognitive_coeff = params["cognitive_coeff"]
    social_coeff = params["social_coeff"]
    inertia = params["inertia"]
    iterations = params["iterations"]
    max_velocity = params.get("max_velocity", 5.0)


    particles = []
    for _ in range(num_particles):
        initial_position = {
            "x": random.uniform(0, 100),
            "y": random.uniform(0, 100)
        }
        
        particles.append({
            "position": initial_position.copy(),
            "velocity": {
                "x": random.uniform(-max_velocity, max_velocity),
                "y": random.uniform(-max_velocity, max_velocity)
            },
            "best_position": initial_position.copy(),
            "best_fitness": float("inf")
        })

    global_best_position = None
    global_best_fitness = float("inf")
    iteration_data = []

    # Main optimization loop
    for it in range(iterations):
        for particle in particles:

            current_fitness = calculate_fitness(particle["position"], goal_x, goal_y)

            # Update personal and global best
            if current_fitness < particle["best_fitness"]:
                particle["best_fitness"] = current_fitness
                particle["best_position"] = particle["position"].copy()

            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle["position"].copy()

            # Stopping criterion: exit if very close to goal
            if global_best_fitness < 0.01:
                break


            r1, r2 = random.random(), random.random()
            
            # Update Velocity
            particle["velocity"]["x"] = (
                inertia * particle["velocity"]["x"] +
                cognitive_coeff * r1 * (particle["best_position"]["x"] - particle["position"]["x"]) +
                social_coeff * r2 * (global_best_position["x"] - particle["position"]["x"])
            )
            
            particle["velocity"]["y"] = (
                inertia * particle["velocity"]["y"] +
                cognitive_coeff * r1 * (particle["best_position"]["y"] - particle["position"]["y"]) +
                social_coeff * r2 * (global_best_position["y"] - particle["position"]["y"])
            )

            # Update position
            particle["position"]["x"] += particle["velocity"]["x"]
            particle["position"]["y"] += particle["velocity"]["y"]

        # Save iteration data
        iteration_data.append({
            "iteration": it,
            "particles": [{"position": p["position"].copy(), "fitness": p["best_fitness"]} for p in particles],
            "global_best": {"position": global_best_position.copy(), "fitness": global_best_fitness}
        })

        # Optional early stopping
        if global_best_fitness < 0.01:
            break

    return {
        "optimal_position": global_best_position,
        "best_fitness": global_best_fitness,
        "iterations": iteration_data
    }

@app.post("/pso")
async def pso_endpoint(request: PSORequest):
    try:
        params = request.dict()
        result = calculate_pso(params)
        return {"status": "success", "data": result}
    except Exception as e:
        print("Error during simulation:", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")