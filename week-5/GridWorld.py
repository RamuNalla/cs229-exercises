import numpy as np

class GridWorld:
    def __init__(self, grid, living_reward=-0.01, goal_reward=1.0, penalty_reward=-1.0, discount_factor = 0.9):
            
            # grid (a matrix with list of list of strings): The grid layout ('.', 'G', 'P'). (.) represents an empty cell, G represents the Goal state, P represents the Penalty state
            # living_reward (float): Reward for moving to a non-terminal state (other than Goal and Penalty).
            # goal_reward (float): Reward for reaching the Goal state.
            # penalty_reward (float): Reward for reaching the Penalty state.
            # discount_factor (float): Discount factor (gamma) for future rewards.

            self.grid = grid
            self.rows = len(grid)
            self.cols = len(grid[0])
            self.living_reward = living_reward        # -0.01
            self.goal_reward = goal_reward            # +1.0
            self.penalty_reward = penalty_reward      # -1.0
            self.discount_factor = discount_factor

            # Actions:
            self.actions = ['up', 'down', 'left', 'right']
            self.action_map = {
                  'up': (-1, 0),
                  'down': (1, 0),
                  'left': (0, -1),
                  'right': (0, 1)
            }

            self.states = []                                        # Stores all valid states the agent can be in
            self.terminal_states = set()                            # Set that will store positions in the grid that are either 'G' or 'P'

            for r in range(self.rows):                              # Loop to go to every cell
                  for c in range(self.cols):
                        if grid[r][c] != "#":                       # Cell is not a wall (the agent can't move in them)
                            self.states.append((r,c))               # Add the valid states to self.states
                            if grid[r][c] == 'G':                   # If the current cell is Goal or Penalty, it is a terminal state
                                self.terminal_states.add((r,c))
                            elif grid[r][c] == 'P':
                                 self.terminal_states.add((r,c))

    def is_terminal(self, state):                                   # Simple helper function to check if a state is either goal or penalty, returns True or False
         return state in self.terminal_states
    

    def get_reward(self, state):                               # Return the reward for reaching a given state
        if state in self.terminal_states:                           # Living_reward is handled separately in the step()
             if self.grid[state[0]][state[1]] == 'G':
                  return self.goal_reward
             elif self.grid[state[0]][state[1]] == 'P':
                  return self.penalty_reward
        
        return 0.0                                                  # If the state is non-terminal, there is no additional reward for being in that state

    # Step function simulates a step in the environment and returns a tuple next_state, reward, done
    # next_state tuple: The state after taking the action
    # reward: The reward received for the transition
    # done: True if the next state is terminal (either G or P), False otherwise
    def step(self, state, action):                                  # Simulates a step in the environment, R
         
        if self.is_terminal(state):                                # If the state is already terminal (G or P), stay there with no additional reward
            return state, 0.0, True 

        dr, dc = self.action_map[action]                           # Return to go up or down in numbers
        next_r, next_c = state[0] + dr, state[1] + dc

        # If the next state is not in the environment, stay in the current state, don't change
        if not (0 <= next_r < self.rows and 0 <= next_c < self.cols): # or self.grid[next_r][next_c] == '#'):
            next_state = state
        else:
            next_state = (next_r, next_c)

        reward = self.living_reward + self.get_reward(next_state)      # living_reward + reward of the next_state
        # If the current state is terminal, it was checked in the first if condition

        done = self.is_terminal(next_state)                             # Check is the next state is terminal

        return next_state, reward, done


                                  