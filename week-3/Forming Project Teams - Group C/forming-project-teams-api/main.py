from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def factorial(num):
    if num == 0 or num == 1:
        return 1
    result = 1
    for i in range(2, num + 1):
        result *= i
    return result


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


@app.get("/teams/")
async def calculate_teams(n: int, k: int):
    if k > n:
        raise HTTPException(status_code=400, detail="k cannot be greater than n")
    if n < 0 or k < 0:
        raise HTTPException(status_code=400, detail="n and k must be non-negative")

    total_teams = comb(n, k)

    binomial_table = [
        {"key": i, **{f"col_{j}": comb(i, j) for j in range(i + 1)}}
        for i in range(n + 1)
    ]

    columns = [{"title": f"Col {j}", "dataIndex": f"col_{j}", "key": f"col_{j}", "align": "center"}
               for j in range(n + 1)]

    return {
        "total_teams": total_teams,
        "binomial_table": binomial_table,
        "columns": columns,
        "n": n,
        "k": k
    }


@app.get("/")
async def root():
    return {"message": "Hello World"}
