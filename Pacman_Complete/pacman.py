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
import pickle
from nodes import Node
from nodes import NodeGroup

class Pacman(Entity):
    def __init__(self, node, pellet_group, nodes, learning, ghosts = None):
        Entity.__init__(self, node)
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)
        
        # Q-learning parameters
        self.q_table = {}
        self.alpha = 0.5  # Learning rate
        self.gamma = 1  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.decay_rate = 0.99  # Decay rate per episode
        self.rng = np.random.default_rng()  # Random number generator
        self.reward = 0 # Reward to be given during learning
        self.learning = learning

        # Reference to the pellets and ghosts
        self.pellets : PelletGroup = pellet_group
        self.ghost_group : GhostGroup = ghosts
        self.nodes : NodeGroup = nodes

        # Holds the previous state and action to use when learning
        self.state = None
        self.prev_dir = self.direction

    def set_epsilon(self, value):
        """
        Sets the epsilon value
        """
        self.epsilon = value

    def decay_epsilon(self):
        """
        Gradually reduces the epsilon / exploration parameter
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

        If the q-value for that state and action does not exist: Create it and set it to 0
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def learn(self, prev_state, action, curr_state):        
        # Calculate q-value, inspired heavily by the function from the exercises
        max_future_reward = 0
        if curr_state:
            next_available_actions = self.validDirections()
            max_future_reward = max([self.get_q_value(curr_state, action) for action in next_available_actions])
        current_q_value = self.get_q_value(prev_state, action)
        self.q_table[(prev_state, action)] = current_q_value + self.alpha * (
            self.reward + self.gamma * max_future_reward - current_q_value
            )
        self.reward = 0

    # Save and load policy, taken from the exercises
    def save_policy(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_policy(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

    def setLearning(self, learn):
        """
        Sets whether or not to learn when the game is running
        """
        self.learning = learn

    def incrementReward(self, value):
        """
        Increments the q-learn reward with value
        """
        self.reward += value

    def getNewState(self):
        """
        Calculates and returns the current state
        """
        # Rewards:
        # Moving to a node : -10, to enourage NOT moving around in circles
        # Getting a pellet : +10
        # Getting a power pellet : +50
        # Killing a ghost : +500
        # Dying to a ghost : -500, High so that we nullify any other positive rewards we got before dying
        # Winning a level : +500

        #### Update State ####
        # Find closest pellet to pacman
        pellets_dists = []
        for pellet in self.pellets.pelletList:
            vec : Vector2 = pellet.position - self.position
            pellets_dists.append(vec.magnitudeSquared())

        for pellet in self.pellets.powerpellets:
            vec : Vector2 = pellet.position - self.position
            pellets_dists.append(vec.magnitudeSquared())
        
        closest_pellet_idx = pellets_dists.index(min(pellets_dists))
        closest_pellet = (self.pellets.pelletList + self.pellets.powerpellets)[closest_pellet_idx]
        closest_pellet = (closest_pellet.position.x, closest_pellet.position.y)

        # Check through the ghost's position and threat level
        ghost_threats = []
        blinky_pos = tuple()
        pinky_pos = tuple()
        inky_pos = tuple()
        clyde_pos = tuple()
        for ghost in self.ghost_group.ghosts:
            ghost_threats.append(ghost.mode.current)
            if ghost.name == BLINKY:
                blinky_pos = (round(ghost.node.position.x), round(ghost.node.position.y)) # We round to only store integer values of positions
            elif ghost.name == PINKY:
                pinky_pos = (round(ghost.node.position.x), round(ghost.node.position.y))
            elif ghost.name == INKY:
                inky_pos = (round(ghost.node.position.x), round(ghost.node.position.y))
            elif ghost.name == CLYDE:
                clyde_pos = (round(ghost.node.position.x), round(ghost.node.position.y))

        threat_level = tuple(ghost_threats)
        pacman_pos = (round(self.node.position.x), round(self.node.position.y))  
        new_state = tuple([pacman_pos, threat_level, closest_pellet, blinky_pos, pinky_pos, inky_pos, clyde_pos])
        ### Finish updating state ###
        return new_state
    
    def setStartState(self):
        """
        Sets the starting state of pacman
        """
        self.state = self.getNewState()

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        # Learn when pacman dies
        if self.learning:
            self.reward -= 500
            self.learn(self.state, self.direction, 0) # Current state is 0, which means max future reward will be 0
        self.alive = False
        self.direction = STOP

    def update(self, dt):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

        if self.overshotTarget():
            # Choose a direction based on the new state and the available directions
            # We do it after the node has been set, so that the available directions are based on the node we just reached
            self.node = self.target
            new_state = self.getNewState()
            direction = self.choose_action(new_state, self.validDirections())

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

            # Learn when reaching a new node
            if self.learning and new_state != self.state and self.prev_dir != STOP:
                self.reward -= 10
                # self.state is the state we had on the previous node
                # self.prev_dir is the direction we took from self.state to get to new_state
                # new_state is the current state
                self.learn(self.state, self.prev_dir, new_state)
            # Update the state attribute to hold the current state
            self.state = new_state
            # Update prev_dir to hold the new direction, that goes to some new node
            self.prev_dir = self.direction

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                # give rewards based on the pellet type
                if pellet.name == POWERPELLET:
                    self.reward += 50
                if pellet.name == PELLET:
                    self.reward += 10
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
