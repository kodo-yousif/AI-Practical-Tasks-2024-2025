import threading
import time
import random
from vacuum_board.vacuum import (
    board, set_board, get_board, get_dirt_pos, get_vacuum_pos, move_to
)

def write_solution(path, total_cost, message=None):
    with open("solution.txt", "w") as f:
        if message:
            f.write(message + "\n")
        else:
            f.write(f"Path: {path}\n")
            f.write(f"Total Cost: {total_cost}\n")

# we are initializing the board on the separate thread because (pygame)
# runs in an event loop, so it blocks other's code from executing
# we do this to let the BFS to run simultaneously
# and we are sleeping it for 2 second to ensure the GUI has initialized
def initialize_board():
    board_thread = threading.Thread(target=board)
    board_thread.start()
    time.sleep(2)
    
    

# BFS algorithm to find the path to the dirt
def bfs_search():
    initial_pos = tuple(get_vacuum_pos())
    goal_pos = tuple(get_dirt_pos())
    board_state = get_board() 
    # this converts the GameBoard into a 2D array of numbers
    # 10 Vacuum
    # 5 Dirt tile
    # 0 Obstacle
    # 1 Floor Tile
    # Ensure this is called after GUI initialization

    # Directions: (movement_name, (i, j offset), cost)
    moves = [
        ("top", (-1, 0), 2),
        ("left", (0, -1), 1),
        ("right", (0, 1), 1),
        ("bottom", (1, 0), 0),
    ]

    # (current_position, path_taken, total_cost)
    queue = [(initial_pos, [], 0)]  
    visited = set([initial_pos])

    while queue:
        (current_pos, path, cost) = queue.pop(0)

        if current_pos == goal_pos:
            return path, cost  # Solution found

        # Explore neighbors
        for move_name, (di, dj), move_cost in moves:
            new_pos = (current_pos[0] + di, current_pos[1] + dj)

            if (
                0 <= new_pos[0] < 6 and
                0 <= new_pos[1] < 6 and
                board_state[new_pos[0]][new_pos[1]] != 0 and  # Not an obstacle
                new_pos not in visited
            ):
                visited.add(new_pos)
                queue.append((new_pos, path + [move_name], cost + move_cost))

    return None, None  # No solution found


def main():

    vacuum_place = random.randint(0, 35)
    dirt_place = random.randint(0, 35)

    while dirt_place == vacuum_place:
        dirt_place = random.randint(0, 35)

    result = []
    while len(result) < 5:
        rand_num = random.randint(0, 35)
        if rand_num != vacuum_place and rand_num != dirt_place:
            result.append(rand_num)

    set_board([vacuum_place, dirt_place, result])
    initialize_board()

    path, total_cost = bfs_search()

    if path is None:
        write_solution([], 0, "No solution found: Obstacles block the path.")
        print("No solution found: Obstacles block the path.")
        return

    write_solution(path, total_cost)

    for move in path:
        # Pause between moves
        time.sleep(0.5)  
        move_to(move)

if __name__ == "__main__":
    main()
