import os
# this is used to find correct paths for assets

import pygame
# library for creating games and visualizing GUIs


# this File is visualizing the squares in your game into 
# vacuum, dirt, floor and obstacles
class Tiles:


	# 10,5,0 or 1 , visual x and y position
	def __init__(self, number, xPos, yPos):
		self.number=number; 
		self.xPos = xPos; 
		self.yPos = yPos;


		# a path for assets directory and images
		assets_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets") 
		image_path = os.path.join(assets_path, f"{self.number}.png")

  
		# loading the image from the selected path and transforming
		# it to fit the rectangle then creating it
		self.image = pygame.image.load(image_path)
		self.image=pygame.transform.scale(self.image,(100,100))
		self.rectangle=self.image.get_rect()
			
   
# there we have to methods since the vacuum moves so we have to update
# the UI of each tile like this --> the previous tile should be reloaded
# with the image of the floor and the current tile to vacuum image 
	def im_vacuum(self): 
		self.number = 10
		assets_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets") 
		image_path = os.path.join(assets_path, "10.png")

		self.image =  pygame.image.load(image_path)
		self.image=pygame.transform.scale(self.image,(100,100))
		self.rectangle=self.image.get_rect()

	def im_floor(self): 
		self.number = 1
		assets_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets") 
		image_path = os.path.join(assets_path, "1.png")
		
		self.image =  pygame.image.load(image_path)
		self.image=pygame.transform.scale(self.image,(100,100))
		self.rectangle=self.image.get_rect()
		
  
