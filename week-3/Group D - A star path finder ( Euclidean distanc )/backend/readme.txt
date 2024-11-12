# A* Pathfinding API with FastAPI

This project provides an A* pathfinding API using FastAPI.

## Requirements

- Python 3.8+
- fastapi and uvicorn libraries

## Setup

1. Clone the repository and navigate to the project folder.
2. Create a virtual environment (optional):
   python -m venv venv
   Activate with:
   - Windows: venv\Scripts\activate
   - macOS/Linux: source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt

## Running the API

To start the FastAPI server, run the following command:

uvicorn main:app --reload

- The server will start on http://127.0.0.1:8000.
- The --reload option enables hot-reloading, so the server automatically reloads when you make code changes.


