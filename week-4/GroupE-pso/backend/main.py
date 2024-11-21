import random
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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


def calculate_pso(params):
    num_particles = params["num_particles"]
    goal_x = params["goal_x"]
    goal_y = params["goal_y"]
    cognitive_coeff = params["cognitive_coeff"]
    social_coeff = params["social_coeff"]
    inertia = params["inertia"]
    iterations = params["iterations"]


    # randomly set positions and velocities for particles
    particles = [
        {
            "position": {"x": random.uniform(0, 100), "y": random.uniform(0, 100)},
            "velocity": {"x": random.uniform(-1, 1), "y": random.uniform(-1, 1)},
            "best_position": None,
            "best_fitness": float("inf")
        }
        for _ in range(num_particles)
    ]

    global_best_position = None
    global_best_fitness = float("inf")
    iteration_data = []

    # Run iterations
    for it in range(iterations):
        for particle in particles:
            
            # square root == ** 0.5
            # Calculates the Euclidean distance between the particle's current position and the goal position (goal_x, goal_y).
            # out fitness function is euclidean distance
            fitness = ((particle["position"]["x"] - goal_x) ** 2 + (particle["position"]["y"] - goal_y) ** 2) ** 0.5

            # Fitness: Smaller distances correspond to better fitness (closer to the goal).
            if fitness < particle["best_fitness"]:
                particle["best_fitness"] = fitness
                particle["best_position"] = particle["position"].copy()

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle["position"].copy()

            # Vnew​=(inertia * Vcurrent)+(cognitive_coeff * r1 * (Pbest − Pcurrent))+(social_coeff * r2​*(Gbest − Pcurrent))
            
            # Update velocity and position
            particle["velocity"]["x"] = (
                inertia * particle["velocity"]["x"] +
                cognitive_coeff * random.random() * (particle["best_position"]["x"] - particle["position"]["x"]) +
                social_coeff * random.random() * (global_best_position["x"] - particle["position"]["x"])
            )
            
            particle["velocity"]["y"] = (
                inertia * particle["velocity"]["y"] +
                cognitive_coeff * random.random() * (particle["best_position"]["y"] - particle["position"]["y"]) +
                social_coeff * random.random() * (global_best_position["y"] - particle["position"]["y"])
            )
            
            particle["position"]["x"] += particle["velocity"]["x"]
            particle["position"]["y"] += particle["velocity"]["y"]

        # Save iteration data
        iteration_data.append({
            "iteration": it,
            "particles": [{"position": p["position"], "fitness": p["best_fitness"]} for p in particles],
            "global_best": {"position": global_best_position, "fitness": global_best_fitness}
        })

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