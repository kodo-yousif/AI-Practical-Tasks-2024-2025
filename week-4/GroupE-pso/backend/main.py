from fastapi import FastAPI
from pydantic import BaseModel
from pso_algorithm import run_pso
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific domains if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PSOParams(BaseModel):
    num_particles: int
    goal_x: float
    goal_y: float
    cognitive_coeff: float
    social_coeff: float
    inertia: float
    iterations: int = 100  # Default value if not provided

@app.post("/start_simulation")
async def start_simulation(params: PSOParams):
    results = run_pso(
        goal_x=params.goal_x,
        goal_y=params.goal_y,
        num_particles=params.num_particles,
        cognitive_coeff=params.cognitive_coeff,
        social_coeff=params.social_coeff,
        inertia=params.inertia,
        iterations=params.iterations
    )
    return {"status": "Simulation complete", "results": results}
