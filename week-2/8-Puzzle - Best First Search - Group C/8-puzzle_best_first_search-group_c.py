import heapq
import copy
import pygame
import time
import threading

import sys
sys.path.append('../')
from puzzle_board.puzzle import board, swapTiles, get_board, init_puzzle, get_random_puzzle
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

WINDOW_WIDTH, WINDOW_HEIGHT = 495, 550
TILE_SIZE = 165
WHITE = (245, 247, 248)
BLACK = (69, 71, 75)
FONT_SIZE = 36
BUTTON_RECT = pygame.Rect(365, 505, 125, 40)

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
    heapq.heappush(priority_queue, (manhattan(start_puzzle), 0, start_puzzle, []))

    while priority_queue:
        _, cost, current_puzzle, path = heapq.heappop(priority_queue)

        if current_puzzle == goal_state:
            return path, cost

        visited.add(tuple(map(tuple, current_puzzle)))

        print(tuple(map(tuple, current_puzzle)))

        for neighbor in get_neighbors(current_puzzle):
            if tuple(map(tuple, neighbor)) not in visited:
                empty_pos = [(i, j) for i in range(3) for j in range(3) if current_puzzle[i][j] == 0][0]
                new_empty_pos = [(i, j) for i in range(3) for j in range(3) if neighbor[i][j] == 0][0]
                # Append the new position of the empty tile to the path
                new_path = path + [new_empty_pos]
                heapq.heappush(priority_queue, (manhattan(neighbor), cost + 1, neighbor, new_path))

    return None, -1


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

def visualize_solution(initial_puzzle, path, total_cost):
    init_puzzle(initial_puzzle)

    board_thread = threading.Thread(target=board)
    board_thread.start()
    time.sleep(2)

    print("Initial Board State:")
    print(get_board())
    for step in path:
        swapTiles(step[0], step[1])
        print(get_board())
        time.sleep(1)

    print("Final Board State (Goal):")
    print(get_board())

    while board_thread.is_alive():
        time.sleep(1)


if __name__ == "__main__":

    initial_puzzle = get_random_puzzle()
    print(initial_puzzle)
    invCount = getInvCount(initial_puzzle)
    if invCount % 2 != 0:
        print("This puzzle is not solvable.")
        exit()
    solution_path, total_cost = best_first_search(initial_puzzle)
    print(solution_path)

    if solution_path:
        visualize_solution(initial_puzzle, solution_path, total_cost)
    else:
        print("No solution found.")
