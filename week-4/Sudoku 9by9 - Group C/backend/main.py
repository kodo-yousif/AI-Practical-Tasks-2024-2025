from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sudoku_ga import run_genetic_algorithm

app = FastAPI(title="Sudoku Solver with Genetic Algorithm (Simplified)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartGARequest(BaseModel):
    board: List[List[int]]
    population_size: Optional[int] = 1000
    generations: Optional[int] = 1000
    mutation_rate: Optional[float] = 0.05
    max_no_improvement: Optional[int] = 100


@app.post("/start_ga")
def start_ga(request: StartGARequest):
    board = request.board
    population_size = request.population_size
    generations = request.generations
    mutation_rate = request.mutation_rate
    max_no_improvement = request.max_no_improvement

    board_np = np.array(board)

    if board_np.shape != (9, 9):
        raise HTTPException(status_code=400, detail="Board must be 9x9")

    if not validate_board(board_np):
        raise HTTPException(status_code=400, detail="Invalid Sudoku board: contains duplicates in fixed cells")

    history_result, graph_data = run_genetic_algorithm(board_np, population_size, generations, mutation_rate, max_no_improvement)

    final_solution = history_result[-1] if history_result else {}
    return {"final_solution": final_solution, "history": history_result, "graph_data": graph_data}


def validate_board(board: np.ndarray) -> bool:
    for i in range(9):
        row = board[i, :]
        if not is_unique(row[row != 0]):
            return False
        column = board[:, i]
        if not is_unique(column[column != 0]):
            return False
    for bi in range(3):
        for bj in range(3):
            block = board[bi * 3:(bi + 1) * 3, bj * 3:(bj + 1) * 3].flatten()
            if not is_unique(block[block != 0]):
                return False
    return True


def is_unique(arr):
    return len(arr) == len(set(arr))
