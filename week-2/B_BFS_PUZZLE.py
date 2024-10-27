# Group - B
# 8-PUZZLE SOLVER USING Breadth-First Search
# Presenter: Muhammad Rajab
import time
import threading
from puzzle_board.puzzle import board, swapTiles, get_board, init_puzzle, get_random_puzzle
from collections import deque  # We use deque for our BFS queue
from tabulate import tabulate


# This class represents a single state of our puzzle
class PuzzleState:
    def __init__(self, board, previous_moves=None, parent=None):
        self.board = board                    # Current board configuration
        self.previous_moves = previous_moves or []  # List of moves made to reach this state
        self.parent = parent                  # Reference to the previous state
    
    # Find where the empty tile (0) is on the board
    def find_empty_tile(self):
        for row in range(3):             # Check each row
            for col in range(3):         # Check each column
                if str(self.board[row][col]) == '0':  # Found the empty tile
                    return row, col
        return None
    
    # Check if we've reached the goal state (winning configuration)
    def is_solved(self):
        goal = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '0']
        ]
        # Convert our current board to strings for comparison
        current = [[str(num) for num in row] for row in self.board]
        return current == goal
    
    # Find all possible moves from current state
    def get_valid_moves(self):
        valid_moves = []
        empty_location = self.find_empty_tile()
        
        if empty_location is None:
            return valid_moves
        
        empty_row, empty_col = empty_location
        
        # Check all four possible moves: up, down, left, right
        possible_moves = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]
        
        # Check which moves are valid (within board boundaries)
        for move_row, move_col in possible_moves:
            new_row = empty_row + move_row
            new_col = empty_col + move_col
            
            # Make sure move stays within the 3x3 board
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                valid_moves.append((new_row, new_col))
                
        return valid_moves
    
    # Create a new state by making a move
    def make_move(self, new_position):
        # Create a copy of the current board
        new_board = [row[:] for row in self.board]
        empty_row, empty_col = self.find_empty_tile()
        new_row, new_col = new_position
        
        # Swap the empty tile with the chosen position
        new_board[empty_row][empty_col], new_board[new_row][new_col] = \
            new_board[new_row][new_col], new_board[empty_row][empty_col]
            
        # Return new state with updated moves list
        return PuzzleState(
            new_board,
            self.previous_moves + [(new_row, new_col)],
            self
        )

# This is our main solving function that implements BFS
def solve_puzzle_bfs(initial_board):
    """
    Breadth-First Search (BFS) Algorithm
    """
    
    # Create our initial state
    start_state = PuzzleState(initial_board)
    
    # If we're lucky and already at solution
    if start_state.is_solved():
        return start_state, 0  # No moves needed
    
    # Create our BFS queue and visited states set
    queue = deque([start_state])     # Queue to hold states we need to check
    visited = set()                  # Keep track of states we've seen
    visited.add(str(initial_board))  # Add initial state to visited
    
    total_moves_explored = 0         # Counter for all moves explored
    
    # Main BFS loop
    while queue:  # While we have states to check
        # Get the next state to check
        current_state = queue.popleft()
        
        # Try each possible move from this state
        for move in current_state.get_valid_moves():
            # Make the move and get new state
            new_state = current_state.make_move(move)
            board_string = str(new_state.board)
            
            # Count this as an explored move
            total_moves_explored += 1
            
            # If we haven't seen this state before
            if board_string not in visited:
                # Check if this is the solution
                if new_state.is_solved():
                    return new_state, total_moves_explored
                # If not solution, add to queue and mark as visited
                visited.add(board_string)
                queue.append(new_state)
    
    # If we get here, no solution was found
    return None, total_moves_explored




# Save the solution steps to a file using tabulate for a nicer format
def save_solution(initial_board, solution_state, total_moves_explored):
    # Open the file with 'utf-8' encoding 'w' means writing-mode it will delete previous content if there's any if we dont need to delete we will use 'a'
    with open('solution.txt', 'w', encoding='utf-8') as file:
        # Write a title for the solution file
        file.write("====== Puzzle Solution ======\n\n")
        
        # Write initial board with clear labels
        file.write("Starting Board:\n")
        file.write(tabulate(initial_board, tablefmt="fancy_grid", headers="firstrow") + "\n")
        
        # Get all states from start to finish
        solution_steps = []
        current = solution_state
        while current:
            solution_steps.append(current)
            current = current.parent
        solution_steps.reverse()
        
        # Writing each step in a nicely formatted table
        file.write("\nSolution Steps:\n")
        for i, state in enumerate(solution_steps[1:], start=1):
            file.write(f"\nStep {i}:\n")
            file.write(tabulate(state.board, tablefmt="fancy_grid", headers="firstrow") + "\n")
        
        # Writing total number of moves in the solution
        total_solution_moves = len(solution_state.previous_moves)
        file.write(f"\nTotal Moves in Solution: {total_solution_moves}\n")
        
        # Writing overall cost, including all explored moves
        file.write(f"Total Moves Explored (Cost): {total_moves_explored}\n")
        file.write("\n===============================\n")




# Main function that puts everything together
def main():
    # Get a random puzzle and initialize the board
    start_board = get_random_puzzle()
    init_puzzle(start_board)
    
    # Start the GUI in a separate thread
    gui_thread = threading.Thread(target=board)
    gui_thread.start()
    time.sleep(2)  # Wait for GUI to start
    
    # Solve the puzzle using BFS
    print("Solving puzzle...")
    solution, total_moves_explored = solve_puzzle_bfs(start_board)
    
    if solution:
        print("Solution found!")
        # Save the solution to file, including total cost
        save_solution(start_board, solution, total_moves_explored)
        
        # Show the solution on the GUI
        print("Showing solution on GUI...")
        for move in solution.previous_moves:
            if not gui_thread.is_alive():
                break
            swapTiles(move[0], move[1])
            time.sleep(1)  # Wait between moves
    else:
        print("No solution found!")
    
    # Wait for GUI to close
    while gui_thread.is_alive():
        time.sleep(1)


# Start the program
main()