import time
import threading
from queue import PriorityQueue
from vacuum_board.vacuum import board, get_board, set_board, get_random_board, move_to, get_dirt_pos, get_vacuum_pos, get_obstacle_pos

def bfs_find_dirt_with_cost_and_obstacles():
    start_pos = tuple(get_vacuum_pos())
    dirt_pos = tuple(get_dirt_pos())

    directions = [(0, 1, 1, 'right'), (1, 0, 0, 'bottom'), (0, -1, 1, 'left'), (-1, 0, 2, 'top')]

    queue = PriorityQueue()
    queue.put((0, start_pos, []))
    visited = set()
    visited.add(start_pos)

    while not queue.empty():
        current_cost, current_pos, path = queue.get()

        if current_pos == dirt_pos:
            return path


        for dx, dy, cost, direction in directions:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)


            if is_valid_position(new_pos) and new_pos not in visited:
                visited.add(new_pos)
                new_cost = current_cost + cost
                queue.put((new_cost, new_pos, path + [direction]))

        print(f"Current Position: {current_pos}, Visited: {visited}")

    return None

def is_valid_position(pos):
    board = get_board()
    rows, cols = len(board), len(board[0])
    obstacle_positions = get_obstacle_pos()

    if 0 <= pos[0] < rows and 0 <= pos[1] < cols and pos not in obstacle_positions:
        return True
    return False

def follow_path(path):
    for move in path:
        move_to(move)
        time.sleep(0.5)

set_board(get_random_board())
board_thread = threading.Thread(target=board)
board_thread.start()

time.sleep(2)

print(get_board())
print(get_vacuum_pos())
print(get_dirt_pos())

path_to_dirt = bfs_find_dirt_with_cost_and_obstacles()
print("test")
print(path_to_dirt)
if path_to_dirt:
    follow_path(path_to_dirt)
    print("Dirt cleaned!")
else:
    print("Vacuum is either trapped or no dirt found.")