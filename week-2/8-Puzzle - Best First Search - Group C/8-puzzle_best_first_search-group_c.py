import heapq
import copy
import pygame
import time
import threading
import sys
from puzzle_board.puzzle import board, swapTiles, get_board, init_puzzle, get_random_puzzle
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def getInvCount(arr):
    inv_count = 0
    flattened_array = [i for row in arr for i in row]
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if flattened_array[j] != 0 and flattened_array[i] != 0 and flattened_array[i] > flattened_array[j]:
                inv_count += 1
    return inv_count

def manhattan(puzzle):
    distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                correct_x = (puzzle[i][j] - 1) // 3
                correct_y = (puzzle[i][j] - 1) % 3
                distance += abs(i - correct_x) + abs(j - correct_y)
    return distance

def get_neighbors(puzzle):
    neighbors = []
    empty_i, empty_j = [(i, j) for i in range(3) for j in range(3) if puzzle[i][j] == 0][0]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_puzzle = copy.deepcopy(puzzle)
            new_puzzle[empty_i][empty_j], new_puzzle[new_i][new_j] = new_puzzle[new_i][new_j], new_puzzle[empty_i][
                empty_j]
            neighbors.append(new_puzzle)
    return neighbors

def best_first_search(start_puzzle):
    priority_queue = []
    visited = set()
    heapq.heappush(priority_queue, (manhattan(start_puzzle), 0, start_puzzle, [], []))

    while priority_queue:
        _, cost, current_puzzle, path, complete_puzzle_path = heapq.heappop(priority_queue)

        if current_puzzle == goal_state:
            return path, cost, complete_puzzle_path + [current_puzzle]

        visited.add(tuple(map(tuple, current_puzzle)))

        for neighbor in get_neighbors(current_puzzle):
            if tuple(map(tuple, neighbor)) not in visited:
                new_empty_pos = [(i, j) for i in range(3) for j in range(3) if neighbor[i][j] == 0][0]
                new_path = path + [new_empty_pos]
                heapq.heappush(priority_queue, (manhattan(neighbor), cost + 1, neighbor, new_path, complete_puzzle_path + [current_puzzle]))

    return None, -1, None

def save_solution_to_file(path, cost, filename='solution.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Initial puzzle state:\n")
        f.write(format_solution_output(path[0]))
        f.write("\nSteps to solution:\n")
        for step_num, state in enumerate(path[1:], 1):
            f.write(f"Step {step_num}:\n")
            f.write(format_solution_output(state))
            f.write("\n")
        f.write(f"Total cost: {cost}\n")

def format_solution_output(state):
    output = "╔═══╦═══╦═══╗\n"
    for row in state:
        output += "║"
        for num in row:
            output += f" {num if num != 0 else ' '} ║"
        output += "\n"
        output += "╠═══╬═══╬═══╣\n"
    output = output[:-14] + "╚═══╩═══╩═══╝"

    return output

def tile_swap_loop():
    global solution_path
    time.sleep(2)
    for step in solution_path:
        print(step)
        swapTiles(step[0], step[1])
        time.sleep(0.5)

def visualize_solution(initial_puzzle):
    init_puzzle(initial_puzzle)

    swap_thread = threading.Thread(target=tile_swap_loop, daemon=True)
    swap_thread.start()

    try:
        board()
    except Exception as e:
        print(f"Board crashed with error: {e}")


if __name__ == "__main__":
    initial_puzzle = get_random_puzzle()
    invCount = getInvCount(initial_puzzle)
    if invCount % 2 != 0:
        print("This puzzle is not solvable\nThe board has an odd number of inversions:", invCount)
        exit()
    solution_path, total_cost, boards = best_first_search(initial_puzzle)
    if solution_path:
        save_solution_to_file(boards, total_cost)
        visualize_solution(initial_puzzle)
    else:
        print("No solution found.")