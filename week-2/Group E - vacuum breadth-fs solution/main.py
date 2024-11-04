import threading
import time
import random
from vacuum_board.vacuum import (board, set_board, get_board, get_dirt_pos, get_vacuum_pos, move_to)

# we are initializing the board on the separate thread because (pygame)
# runs in an event loop, so it blocks other's code from executing
# we do this to let the BFS to run simultaneously
# and we are sleeping it for 2 second to ensure the GUI has initialized
def initialize_board():
    board_thread = threading.Thread(target=board)
    board_thread.start()
    time.sleep(2)
    
    
def write_solution(path, total_cost, message=None):
    with open("solution.txt", "w") as f:
        f.write("Initial Board State:\n")
        for row in get_board():
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")
        
        if message:
            f.write(message + "\n")
        else:
            f.write(f"Path: {path}\n")
            f.write(f"Total Cost: {total_cost}\n")

# BFS algorithm to find the shortest path to the dirt (queue)
def bfs_search():
    initial_pos = tuple(get_vacuum_pos()) #(1,1)
    goal_pos = tuple(get_dirt_pos())      #(2,3)
    board_state = get_board()             # 2d array

    # Directions: (movement_name, (i, j offset), cost)
    moves = [
        ("top", (-1, 0), 2),
        ("left", (0, -1), 1),
        ("right", (0, 1), 1),
        ("bottom", (1, 0), 0),
    ]

    # (current_position, path_taken, total_cost)
    queue = [(initial_pos, [], 0)] # (1,1), no path ,0
    visited = set([initial_pos])   # add (1,1) to (visited set) to prevent visiting again

    while queue:
        (current_pos, path, cost) = queue.pop(0)   
        # (1,1) , []       , 0 first
        # (0, 1), ["top"]  , 2 second
        # (1, 2), ["right"], 1 third
        
        if current_pos == goal_pos: # if pass then that means Solution is found
            return path, cost  

        # Explore neighbors to check the valid moves
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

    return None, None  # if No solution found so return this


def main():
    
    set_board()
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
