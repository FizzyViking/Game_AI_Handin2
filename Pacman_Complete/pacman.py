import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
# new imports
import numpy as np # Needed for utility functions
from pellets import PelletGroup # Import pelletgroup to get pellets positions
from ghosts import GhostGroup

class Pacman(Entity):
    def __init__(self, node, pellet_group, ghosts = None):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)
        
        # Q-learning parameters
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.decay_rate = 0.99  # Decay rate per episode
        self.rng = np.random.default_rng()  # Random number generator
        self.reward = 0 # Reward to be given during learning

        self.state = None

        # Reference to the pellets and ghosts
        self.pellets : PelletGroup = pellet_group
        self.ghost_group : GhostGroup = ghosts

    def decay_epsilon(self):
        """
        Decays the epsilon value exponentially after each episode.
        """
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)

    def choose_action(self, state, available_actions):
        """
        Returns an action based on the given state and available actions
        """
        # Generate a random float between 0.0 and 1.0 to determine whether or not to explore based on the episilon parameter
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)
        else:
            # Choose the action with the highest Q-value for the current state
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q_value = max(q_values)
            # In case of multiple max values, randomly select one
            max_indices = [i for i, q in enumerate(q_values) if q == max_q_value]
            return available_actions[self.rng.choice(max_indices)]

    def get_q_value(self, state, action):
        """
        Returns the q-value in q table for the given state and action
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def learn(self, state, action, next_state):

        # Capture state with pacmans pos, nearest pellet and ghost's positions and their threat level (flee or not)
        # Is pacman powered or not? (can kill ghosts)
        # Rewards:
        # Moving to a node : -10, to enourage moving around in circles
        # Getting a power pellet : +10
        # Killing a ghost : +100
        # Dying to a ghost : -100
        # Winning a level : +500
        # Losing a level or dying: -500

        # Find closest pellet to pacman
        pellets_dists = []
        for pellet in self.pellets.pelletList:
            vec : Vector2 = pellet.position - self.position
            pellets_dists.append(vec.magnitudeSquared())
        for pellet in self.pellets.powerpellets:
            vec : Vector2 = pellet.position - self.position
            pellets_dists.append(vec.magnitudeSquared())
        closest_pellet = min(pellets_dists)

        # Check through the ghosts and their position and threat level
        threat_level = 0
        blinky_pos = Vector2()
        pinky_pos = Vector2()
        inky_pos = Vector2()
        clyde_pos = Vector2()
        for ghost in self.ghost_group.ghosts:
            if ghost.name == BLINKY:
                blinky_pos = ghost.position
            elif ghost.name == PINKY:
                pinky_pos = ghost.position
            elif ghost.name == INKY:
                inky_pos = ghost.position
            elif ghost.name == CLYDE:
                clyde_pos = ghost.position
            if ghost.mode.current == CHASE:
                threat_level = 1
        
        next_available_actions = STOP
        max_future_reward = max([self.q_table.get((next_state, action)) for action in next_available_actions])
        current_q_value = self.get_q_value(state, action)
        self.q_table[(state, action)] = current_q_value + self.alpha * (
            self.reward + self.gamma * max_future_reward - current_q_value
            )

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP
        # Learn when pacman dies
        self.reward -= 500
        self.learn(self.state, self.direction, 0)

    def update(self, dt):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

        # Learn when reaching a new node
        # Choose a direction based on the state and the available directions
        direction = self.choose_action(self.state, self.validDirections())
        if self.target == self.node:
            self.learn(self.state, direction, 0)

        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()  

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
