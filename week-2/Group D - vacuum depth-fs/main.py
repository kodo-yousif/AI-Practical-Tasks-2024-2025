import time
import threading
from vacuum_board.vacuum import board, set_board, get_dirt_pos, get_vacuum_pos, move_to, puzzle, GameBoard

# Define movement costs and directions
moves = {'top': (-1, 0, 2), 'left': (0, -1, 1), 'right': (0, 1, 1), 'bottom': (1, 0, 0)}
# Initialize the board
set_board()
initial_board = [row[:] for row in puzzle]
vacuum_pos, dirt_pos = get_vacuum_pos(), get_dirt_pos()

# Depth-First Search function
def dfs(pos, path, cost, visited):
    if pos == dirt_pos:
        return path, cost
    visited.add(tuple(pos))

    for name, (dx, dy, move_cost) in moves.items():
        new_pos = [pos[0] + dx, pos[1] + dy]
        if 0 <= new_pos[0] < 6 and 0 <= new_pos[1] < 6 and initial_board[new_pos[0]][new_pos[1]] != 0 and tuple(new_pos) not in visited:
            result = dfs(new_pos, path + [name], cost + move_cost, visited)
            if result:
                return result
    visited.remove(tuple(pos))
    return None

# Save file
result = dfs(vacuum_pos, [], 0, set())
with open('solution.txt', 'w') as file:
    # Format the initial board with better readability
    file.write("Initial Board:\n")
    for row in initial_board:
        formatted_row = ' | '.join(map(str, row))  # Adding separators between columns
        file.write(f"{formatted_row}\n")
    file.write("\n")

    if result:
        path, total_cost = result
        file.write("Steps to solution with costs:\n")
        # Write each step with its cost
        step_details = []
        for move in path:
            cost = moves[move][2]  # Get the cost of each move
            step_details.append(f"{move} (cost: {cost})")
        file.write(" -> ".join(step_details) + f"\n\nOverall cost: {total_cost}\n")
    else:
        file.write("No solution because of obstacles.\n")



def move_to_with_dirt(direction):
    from vacuum_board.vacuum import vacuumTilePosI, vacuumTilePosJ
    temp_i, temp_j = vacuumTilePosI + moves[direction][0], vacuumTilePosJ + moves[direction][1]
    if 0 <= temp_i < 6 and 0 <= temp_j < 6 and GameBoard[temp_i][temp_j].number != 0:
        if [temp_i, temp_j] == dirt_pos:
            GameBoard[temp_i][temp_j].number = 1
        move_to(direction)
        if [temp_i, temp_j] == dirt_pos:
            GameBoard[temp_i][temp_j].number = 5

def execute_moves(moves_list):
    time.sleep(1)
    for move in moves_list:
        time.sleep(0.5)
        move_to_with_dirt(move)
    print("Vacuum has reached the dirt and cleaned it.")

board_thread = threading.Thread(target=board)
board_thread.start()
if result:
    execute_moves(result[0])
else:
    print("No solution found due to obstacles.")
while board_thread.is_alive():
    time.sleep(0.1)
