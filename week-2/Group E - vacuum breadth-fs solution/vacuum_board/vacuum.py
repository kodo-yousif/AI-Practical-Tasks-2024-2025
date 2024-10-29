import math
import pygame
import random 
from vacuum_board.Tiles import Tiles

# global variables
GameBoard = [[None for _ in range(6)] for _ in range(6)]
vacuumTilePosI=0 
vacuumTilePosJ=0
dirtTilePosI=0 
dirtTilePosJ=0
init_board = [[1 for _ in range(6)] for _ in range(6)]
puzzle = init_board

# randomly generates 6x6 board 
def get_random_board(vacuum = False, dirt = False, obstacles = False ):

	if vacuum: 
		num1 = vacuum
	else:
		num1 = random.randint(0, 35)

	if vacuum: 
		num2 = dirt
	else:
		num2 = random.randint(0, 35)

	while num1 == num2:
		num2 = random.randint(0, 35)

	if obstacles:
		obs = obstacles
	else:
		len = random.randint(1, 10)
		obs = [1 for _ in range(len)]
		for i in range(len):
			rand = random.randint(0, 35)

			while rand == num1 or rand == num2:
				rand = random.randint(0, 35)
			
			obs[i] = rand

	return [num1 , num2, obs]


# initializes the game board
def set_board(random_board = get_random_board()):
    global puzzle, vacuumTilePosI, vacuumTilePosJ, dirtTilePosI, dirtTilePosJ

    vacuum = random_board[0]
    dirt = random_board[1]
    obstacles = random_board[2]

    # "//" to indicate which row 
    # "%" to indicate which column 
    # 6 since it's 6x6 board
    vacuumTilePosI = vacuum // 6 
    vacuumTilePosJ = vacuum % 6

    dirtTilePosI = dirt // 6
    dirtTilePosJ = dirt % 6

    # Initialize puzzle with floor tiles (value 1)
    puzzle = [[1 for _ in range(6)] for _ in range(6)]

    # Place vacuum (value 10)
    puzzle[vacuumTilePosI][vacuumTilePosJ] = 10

    # Place dirt (value 5)
    puzzle[dirtTilePosI][dirtTilePosJ] = 5

    # Place obstacles (value 0)
    for obs in obstacles:
        obs_i, obs_j = obs // 6, obs % 6
        puzzle[obs_i][obs_j] = 0

# this renders the game board using (pygame)
def board():
    global GameBoard, puzzle

    pygame.init()
    bg = (200, 200, 200) # background of the displayed window
    gameDisplay = pygame.display.set_mode((670, 670)) # size of the displayed window
    pygame.display.set_caption('Vacuum Board') # title of the displayed window

    tilePosX, tilePosY = 10, 10 


    for i in range(6):
        for j in range(6):
            GameBoard[i][j] = Tiles(puzzle[i][j], tilePosX, tilePosY)
            tilePosX += 110
        tilePosX = 10
        tilePosY += 110

    clock = pygame.time.Clock()
    gameExit = False

    while not gameExit:
        gameDisplay.fill(bg)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
                pygame.quit()
                quit()

        for i in range(6):
            for j in range(6):
                gameDisplay.blit(GameBoard[i][j].image, (GameBoard[i][j].xPos, GameBoard[i][j].yPos))

        pygame.display.update()
        clock.tick(30)

# this moves the vacuum to top,bottom,left,right if the move is valid
def move_to(place):
    global GameBoard, vacuumTilePosI, vacuumTilePosJ

    tempI, tempJ = vacuumTilePosI, vacuumTilePosJ

    # there we are changing the indexes of vacuum position
    if place == "top":
        tempI -= 1
    elif place == "bottom":
        tempI += 1
    elif place == "right":
        tempJ += 1
    elif place == "left":
        tempJ -= 1
    else:
        print("Error: Invalid action.")
        return get_board()

    # checks the vacuum doesn't go outside of the board
    if not (0 <= tempI < 6 and 0 <= tempJ < 6):
        print("Error: Can't go out of the board.")
        return get_board()


    # checks the obstacle around the vacuum
    if GameBoard[tempI][tempJ].number == 0:
        print("Error: Can't go to a blocked tile.")
        return get_board()


    GameBoard[tempI][tempJ].im_vacuum()
    GameBoard[vacuumTilePosI][vacuumTilePosJ].im_floor()

    vacuumTilePosI, vacuumTilePosJ = tempI, tempJ


    # Refresh the display to show movement
    pygame.display.update()
    return get_board()


def get_dirt_pos():
	return [dirtTilePosI, dirtTilePosJ]

def get_vacuum_pos():
	return [vacuumTilePosI, vacuumTilePosJ]


# returns the current states of each rectangle of the 6x6 board as 2D array 
# according to their values 0,1,5,10
def get_board():
    global GameBoard
    array_6x6 = [[0 for _ in range(6)] for _ in range(6)]

    for i in range(6):
        for j in range(6):
            if isinstance(GameBoard[i][j], Tiles):
                array_6x6[i][j] = GameBoard[i][j].number
            else:
                print(f"Error: GameBoard[{i}][{j}] is not a Tiles object.")
                raise ValueError("GameBoard contains non-Tiles objects.")

    return array_6x6