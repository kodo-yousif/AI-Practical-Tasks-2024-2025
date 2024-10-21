import heapq
import copy
import pygame
import time

from puzzle_board.puzzle import get_random_puzzle
from puzzle_board.Tiles import Tiles
print(get_random_puzzle())
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

WINDOW_WIDTH, WINDOW_HEIGHT = 495, 550
TILE_SIZE = 165
WHITE = (245, 247, 248)
BLACK = (69, 71, 75)
FONT_SIZE = 36
BUTTON_RECT = pygame.Rect(365, 505, 125, 40)

def manhattan(puzzle):
    distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                correct_x = (puzzle[i][j] - 1) // 3
                correct_y = (puzzle[i][j] - 1) % 3
                distance += abs(i - correct_x) + abs(j - correct_y)
                print(distance, correct_x, correct_y, i, j)
    print('|')
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
        output +=           "╠═══╬═══╬═══╣\n"
    output = output[:-14] + "╚═══╩═══╩═══╝"
    return output


def visualize_solution(path, total_cost):
    pygame.init()
    gameDisplay = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('8-Puzzle Solution')

    font = pygame.font.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    BUTTON_RECT = pygame.Rect(365, 505, 125, 40)
    button_text = font.render('Try again', True, BLACK)

    try_again = False

    for step_num, state in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if BUTTON_RECT.collidepoint(event.pos) and step_num == len(path) - 1:
                    try_again = True
                    if __name__ == "__main__":
                        initial_puzzle = get_random_puzzle()
                        solution_path, total_cost = best_first_search(initial_puzzle)

                        if solution_path:
                            save_solution_to_file(solution_path, total_cost)
                            visualize_solution(solution_path, total_cost)
                        else:
                            print("No solution found.")

        gameDisplay.fill(BLACK)

        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    tile = Tiles(state[i][j], TILE_SIZE * j, TILE_SIZE * i)
                    gameDisplay.blit(pygame.transform.scale(tile.image, (TILE_SIZE, TILE_SIZE)), tile.rectangle)

        if step_num == len(path) - 1:
            total_cost_text = font.render(f'Total cost: {total_cost}', True, WHITE)
            gameDisplay.blit(total_cost_text, (10, 510))
        else:
            step_text = font.render(f'Step: {step_num + 1}', True, WHITE)
            gameDisplay.blit(step_text, (10, 510))

        pygame.draw.rect(gameDisplay, WHITE, BUTTON_RECT, border_radius=5)
        gameDisplay.blit(button_text, (BUTTON_RECT.x + 10, BUTTON_RECT.y + 5))

        pygame.display.update()
        time.sleep(1)
        clock.tick(1)

    while not try_again:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if BUTTON_RECT.collidepoint(event.pos):
                    try_again = True
                    if __name__ == "__main__":

                        initial_puzzle = get_random_puzzle()
                        solution_path, total_cost = best_first_search(initial_puzzle)

                        if solution_path:
                            save_solution_to_file(solution_path, total_cost)
                            visualize_solution(solution_path, total_cost)
                        else:
                            print("No solution found.")

        gameDisplay.fill(BLACK)
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    tile = Tiles(state[i][j], TILE_SIZE * j, TILE_SIZE * i)
                    gameDisplay.blit(pygame.transform.scale(tile.image, (TILE_SIZE, TILE_SIZE)), tile.rectangle)

        total_cost_text = font.render(f'Total cost: {total_cost}', True, WHITE)
        gameDisplay.blit(total_cost_text, (10, 510))

        pygame.draw.rect(gameDisplay, WHITE, BUTTON_RECT, border_radius=5)
        gameDisplay.blit(button_text, (BUTTON_RECT.x + 10, BUTTON_RECT.y + 5))

        pygame.display.update()
        clock.tick(30)

    pygame.quit()





if __name__ == "__main__":

    initial_puzzle = get_random_puzzle()
    solution_path, total_cost = best_first_search(initial_puzzle)

    if solution_path:
        save_solution_to_file(solution_path, total_cost)
        visualize_solution(solution_path, total_cost)
    else:
        print("No solution found.")
