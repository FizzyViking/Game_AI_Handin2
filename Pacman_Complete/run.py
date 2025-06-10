import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.episodes = 0 # Episodes here to keep track of it during learning
        self.speedModifier = 1 # Speed modifier to speed up the game
        self.episilon = 0.9 # qlearn parameter
        self.learning = False # whether to learn or not
        self.runUntilWin = False

    def setEpisodes(self, episodes):
        self.episodes = episodes

    def setLearning(self, learn):
        self.learning = learn

    def setEpsilon(self, value):
        self.episilon = value
    
    def setRunUntilWin(self, value):
        self.runUntilWin = value

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart), self.pellets, self.nodes, self.learning) # Edited to give pacman reference to the pellets and ghosts, and set whether to learn
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.pacman.ghost_group = self.ghosts

        # Set pacman speed modifier, epsilon, and start state
        self.pacman.speedModifier = self.speedModifier
        self.pacman.set_epsilon(self.episilon)
        self.pacman.setSpeed(100)
        self.pacman.setStartState()
        
        # Set the ghost's speed modifier
        for ghost  in self.ghosts.ghosts:
            ghost.speedModifier = self.speedModifier
            ghost.mode.speedModifier = self.speedModifier
            ghost.setSpeed(100)
        
        # Load the policy between resets of pacman, so that he has the newest q-table available
        self.pacman.load_policy("policies/policy2.pkl")

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def startGame_old(self):      
        self.mazedata.loadMaze(self.level)#######
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

        

    def update(self):
        dt = self.clock.tick(30) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.incrementReward(500)
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            #self.textgroup.showText(GAMEOVERTXT)
                            #self.pause.setPause(pauseTime=3, func=self.restartGame)
                            self.restartGame()
                        else:
                            #self.pause.setPause(pauseTime=3, func=self.resetLevel)
                            self.resetLevel()
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                #print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        print(f"OMG COMPLETED LVL {self.level}") # Just to notify whether a level was completed or not
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.pacman.incrementReward(500)
        self.pacman.learn(self.pacman.state, self.pacman.direction, 0)
        self.startGame()
        self.textgroup.updateLevel(self.level)
        self.pause.paused = False

    def restartGame(self):
        self.lives = 5
        self.level = 0
        self.pause.paused = True
        self.fruit = None

        # If we are learning, save the policy and decrease the epsilon parameter
        # startGame() resets the pacman object, and thus we have to save the policy and load it again after each episode
        if self.pacman.learning:
            self.pacman.save_policy("policies/policy2.pkl") 
            self.pacman.decay_epsilon() # Decrease epsilon after each episode
            self.episilon = self.pacman.epsilon

        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []
        if self.episodes > 0: # If there are episodes left, restart the game
            self.episodes -= 1 # Decrement episodes after a game
            print("RESTARTING, EPISODES LEFT: ", self.episodes)
            self.textgroup.hideText()
            self.pacman.set_epsilon(self.episilon)
            print("EXPLORATION: ", self.pacman.epsilon)
            self.pause.paused = False # Unpause game automatically
        elif self.level == 0 and self.runUntilWin:
            self.textgroup.hideText()
            self.pause.paused = False # Unpause game automatically
        elif self.episodes == 0: # Otherwise, quit
            exit()

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)
        self.textgroup.hideText()
        self.pause.paused = False # Unpause game automatically 

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    speedModifier = 3 # Speed modifier to accelerate the speed of the enities
    game.speedModifier = speedModifier
    
    learning = True
    runUntilWin = False
    learnAndUsePolicy = True
    episodes = 50
    game.setLearning(learning)
    game.setRunUntilWin(runUntilWin)

    if not learning: game.setEpsilon(0.0) # If we're not learning, set the exploration parameter to 0, so we only use actions based on the learned q-values
    game.startGame()

    # If we are learning
    if learning:
        game.setEpisodes(episodes)
        if learnAndUsePolicy: # If we are learning based on a previously learned policy
            game.pacman.load_policy("policies/policy2.pkl")
    else:
        game.pacman.load_policy("policies/policy2.pkl")

    while True:
        game.update()
