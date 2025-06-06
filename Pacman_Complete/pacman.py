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
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.decay_rate = 0.99  # Decay rate per episode
        self.rng = np.random.default_rng()  # Random number generator
        self.reward = 0 # Reward to be given during learning
        self.learning = learning
        self.state = None

        # Reference to the pellets and ghosts
        self.pellets : PelletGroup = pellet_group
        self.ghost_group : GhostGroup = ghosts
        self.nodes : NodeGroup = nodes

    def set_epsilon(self, value):
        """
        Sets the epsilon value
        """
        self.epsilon = value

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
            print(f"Going {max_indices} with Q-Value: {max_q_value}")
            return available_actions[self.rng.choice(max_indices)]

    def get_q_value(self, state, action):
        """
        Returns the q-value in q table for the given state and action

        If the q-value for that state and action does not exist: Create it and set it to 0
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def learn(self, state, action, next_state):

        # Capture state with pacmans pos, nearest pellet and ghost's positions and their threat level (flee or not)
        # Is pacman powered or not? (can kill ghosts)
        # Rewards:
        # Moving to a node : -10, to enourage NOT moving around in circles
        # Getting a pellet : +10
        # Getting a power pellet : +50
        # Killing a ghost : +500
        # Dying to a ghost : -100
        # Winning a level : +500
        # Losing a level or dying: -500

        # Find the available directions for pacmans target node (Node he's moving towards)
        max_future_reward = 0
        if next_state:
            next_available_actions = self.get_directions_for_target(self.nodes.getNodeFromPixels(next_state[1][0], next_state[1][1]))
            max_future_reward = max([self.get_q_value(next_state, action) for action in next_available_actions])
        current_q_value = self.get_q_value(state, action)
        self.q_table[(state, action)] = current_q_value + self.alpha * (
            self.reward + self.gamma * max_future_reward - current_q_value
            )
        #print(f"Updating Q-Table with state: {state}, action: {action}, Q-Value: {self.get_q_value(state, action)}")
        #print(f"Learning with reward:{self.reward}")
        self.reward = 0

    def get_directions_for_target(self, target : Node):
        """
        Return available directions for the given target. Used for getting the next state
        """
        actions = []
        for direction in self.directions:
            if direction is not STOP and self.name in target.access[direction]:
                    if target.neighbors[direction] is not None:
                        actions.append(direction)
        
        return actions

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
        if self.learning:
            self.reward -= 500
            self.learn(self.state, self.direction, 0)

    def update(self, dt):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

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

        # Check through the ghosts and their position and threat level
        ghost_threats = []
        blinky_pos = tuple()
        pinky_pos = tuple()
        inky_pos = tuple()
        clyde_pos = tuple()
        for ghost in self.ghost_group.ghosts:
            ghost_threats.append(ghost.mode.current)
            if ghost.name == BLINKY:
                blinky_pos = (round(ghost.position.x), round(ghost.position.y))
            elif ghost.name == PINKY:
                pinky_pos = (round(ghost.position.x), round(ghost.position.y))
            elif ghost.name == INKY:
                inky_pos = (round(ghost.position.x), round(ghost.position.y))
            elif ghost.name == CLYDE:
                clyde_pos = (round(ghost.position.x), round(ghost.position.y))

        threat_level = tuple(ghost_threats)
        pacman_pos = (round(self.position.x), round(self.position.y))
        pacman_target = (self.target.position.x, self.target.position.y)
        new_state = tuple([pacman_pos, pacman_target, threat_level, closest_pellet, blinky_pos, pinky_pos, inky_pos, clyde_pos])
        self.state = new_state
        ### Finish updating state ###

        # Choose a direction based on the state and the available directions
        direction = self.choose_action(self.state, self.validDirections())

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

            # Learn when reaching a new node
            if self.learning:
                self.reward -= 10
                #pacman_pos = (round(self.position.x), round(self.position.y))
                pacman_target = (self.target.position.x, self.target.position.y)
                #estimate_next_state = tuple([pacman_pos, pacman_target, threat_level, closest_pellet, blinky_pos, pinky_pos, inky_pos, clyde_pos])
                blinky_pos_future = (self.ghost_group.ghosts[0].target.position.x, self.ghost_group.ghosts[0].target.position.y)
                pinky_pos_future = (self.ghost_group.ghosts[1].target.position.x, self.ghost_group.ghosts[1].target.position.y)
                inky_pos_future = (self.ghost_group.ghosts[2].target.position.x, self.ghost_group.ghosts[2].target.position.y)
                clyde_pos_future = (self.ghost_group.ghosts[3].target.position.x, self.ghost_group.ghosts[3].target.position.y)
                estimate_next_state = tuple([pacman_pos, pacman_target, threat_level, closest_pellet, blinky_pos_future, pinky_pos_future, inky_pos_future, clyde_pos_future])
                self.learn(self.state, direction, estimate_next_state)
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
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
