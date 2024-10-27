import time
import threading
from puzzle_board.puzzle import init_puzzle, board, swapTiles, get_random_puzzle


class EnhancedPuzzleSolver:
    def __init__(self):
        # Define the goal state of the puzzle
        self.goal_state = [[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 0]]  # 0 represents the empty space
        self.max_depth = 100  # Limit for depth in DFS to prevent infinite loops

    def is_solvable(self, puzzle):
        """Check if the puzzle is solvable by counting inversions.

        An inversion is when a higher-numbered tile precedes a lower-numbered tile.
        The number of inversions in a solvable 8-puzzle must be even.
        """
        # Flatten the puzzle into a single list
        flat_puzzle = [num for row in puzzle for num in row]
        inversions = 0  # Initialize inversion count

        # Count inversions
        for i in range(len(flat_puzzle)):
            for j in range(i + 1, len(flat_puzzle)):
                # Skip the empty tile represented by 0
                if flat_puzzle[i] != 0 and flat_puzzle[j] != 0 and flat_puzzle[i] > flat_puzzle[j]:
                    inversions += 1

        # If inversions count is even, the puzzle is solvable
        return inversions % 2 == 0

    def solve_puzzle(self, start_state):
        """Solve the puzzle using Depth-First Search (DFS)."""
        # Initialize the stack with the starting state
        stack = [(start_state, [], 0)]
        visited = set()  # Keep track of visited states to prevent cycles

        while stack:
            # Pop the last state from the stack (LIFO order for DFS)
            current_state, moves, depth = stack.pop()

            # If current state is the goal state, return the moves taken to reach it
            if current_state == self.goal_state:
                return moves

            # Convert current state to a tuple of tuples to make it hashable for the visited set
            state_tuple = tuple(tuple(row) for row in current_state)

            # If we've already visited this state or exceeded max depth, skip it
            if state_tuple in visited or depth >= self.max_depth:
                continue

            # Mark the current state as visited
            visited.add(state_tuple)

            # Find the position of the empty space (0)
            empty_pos = self.find_empty_space(current_state)

            # Get all possible moves from the current empty position
            possible_moves = self.get_possible_moves(empty_pos)

            # For each possible move, generate the new state and add it to the stack
            for move_pos in possible_moves:
                # Create a new state by swapping the empty tile with the adjacent tile
                new_state = [row[:] for row in current_state]  # Create a deep copy of the current state
                new_state[empty_pos[0]][empty_pos[1]], new_state[move_pos[0]][move_pos[1]] = \
                    new_state[move_pos[0]][move_pos[1]], new_state[empty_pos[0]][empty_pos[1]]  # Swap

                # Append the new state to the stack with updated moves and depth
                stack.append((new_state, moves + [move_pos], depth + 1))

        # If no solution is found within max_depth, return None
        return None

    def find_empty_space(self, puzzle):
        """Find the coordinates of the empty space (0) in the puzzle."""
        for row in range(3):
            for col in range(3):
                if puzzle[row][col] == 0:
                    return row, col  # Return the position as a tuple (row, col)
        # If empty space not found (should not happen), return invalid position
        return -1, -1

    def get_possible_moves(self, empty_pos):
        """Get valid positions we can move to from the empty space.

        The empty space can move up, down, left, or right if within bounds.
        """
        row, col = empty_pos
        # Define possible directions: up, down, left, right
        directions = [(-1, 0),  # Up
                      (1, 0),  # Down
                      (0, -1),  # Left
                      (0, 1)]  # Right

        # Calculate new positions and filter out positions that are out of bounds
        possible_moves = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # Check if new position is within the puzzle bounds
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                possible_moves.append((new_row, new_col))

        return possible_moves

def solve_and_display():
    """Main function to solve and display the puzzle solution."""
    # Create an instance of the puzzle solver
    solver = EnhancedPuzzleSolver()

    # Continuously get random puzzles until a solvable one is found
    solvable_puzzle = get_random_puzzle()
    while not solver.is_solvable(solvable_puzzle):
        solvable_puzzle = get_random_puzzle()

    # Initialize the puzzle board with the solvable puzzle
    init_puzzle(solvable_puzzle)

    # Start the puzzle display in a separate thread
    display_thread = threading.Thread(target=board)
    display_thread.daemon = True  # Daemonize thread to close it when main program exits
    display_thread.start()

    # Wait briefly to ensure the display is initialized
    time.sleep(2)

    try:
        # Print and write the starting puzzle state to a file
        print("\nStarting puzzle state:")
        with open("solution.txt", "w") as f:
            # Write the puzzle state to the file
            for row in solvable_puzzle:
                puzzle_row = " ".join(map(str, row))
                print(puzzle_row)
                f.write(puzzle_row + "\n")
            f.write("\nSolving puzzle using DFS...\n")

        # Solve the puzzle using DFS
        solution = solver.solve_puzzle(solvable_puzzle)

        if solution:
            print(f"\nFound solution in {len(solution)} moves!")
            with open("solution.txt", "a") as f:
                # Apply each move in the solution
                for move in solution:
                    time.sleep(0.25)  # Slight delay for display purposes
                    print(f"Moving empty space to position: {move}")
                    swapTiles(*move)  # Swap tiles on the display
                    f.write(f"Move empty space to: {move}\n")
                f.write(f"\nTotal moves: {len(solution)}\n")
        else:
            print("\nNo solution found! The puzzle may be too complex for DFS.")
            print("Try running the program again or consider using BFS instead.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Try running the program again.")


if __name__ == "__main__":
    solve_and_display()
