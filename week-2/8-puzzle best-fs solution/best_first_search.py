import heapq
import copy
import pygame
import time

from puzzle_board.puzzle import get_random_puzzle
from puzzle_board.Tiles import Tiles

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


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
            return path + [current_puzzle], cost

        visited.add(tuple(map(tuple, current_puzzle)))

        for neighbor in get_neighbors(current_puzzle):
            if tuple(map(tuple, neighbor)) not in visited:
                heapq.heappush(priority_queue, (manhattan(neighbor), cost + 1, neighbor, path + [current_puzzle]))

    return None, -1


def save_solution_to_file(path, cost, filename='solution.txt'):
    with open(filename, 'w') as f:
        f.write("Initial puzzle state:\n")
        for row in path[0]:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("\nSteps to solution:\n")
        for step_num, state in enumerate(path[1:], 1):
            f.write(f"Step {step_num}:\n")
            for row in state:
                f.write(' '.join(map(str, row)) + '\n')
            f.write("\n")
        f.write(f"Total cost: {cost}\n")


def visualize_solution(path):
    pygame.init()
    gameDisplay = pygame.display.set_mode((595, 595))
    pygame.display.set_caption('8-Puzzle Solution')

    white = (255, 255, 255)
    clock = pygame.time.Clock()

    for state in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        gameDisplay.fill(white)
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    tile = Tiles(state[i][j], 200 * j + 10, 200 * i + 10)
                    gameDisplay.blit(tile.image, tile.rectangle)
        pygame.display.update()
        time.sleep(1)
        clock.tick(1)

    pygame.quit()


if __name__ == "__main__":
    initial_puzzle = get_random_puzzle()
    solution_path, total_cost = best_first_search(initial_puzzle)

    if solution_path:
        save_solution_to_file(solution_path, total_cost)
        visualize_solution(solution_path)
    else:
        print("No solution found.")
