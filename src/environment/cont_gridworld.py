import math
from copy import deepcopy
import numpy as np

class ContinuousGridWorld:
    """
    Initialize the ContinuousGridWorld environment.

    Args:
        grid (undefined): A grid representing the environment layout.
        wind (undefined): Wind parameters affecting the agent's movement.
        slip_prob (undefined): Probability of slipping during movement.
        verbose (bool, optional): Whether to print verbose information.
    """
    def __init__(self, grid=None, wind=None, slip_prob=None, verbose=False):
        self.verbose = verbose
        self.max_row = 10
        self.max_col = 10
        self.initial_pos = [2, 2]  # Starting position (row, col)
        self.goal = self.terminal = [[7, 7], [7, 8], [8, 7], [8, 8]]
        self.trap = [[2, 4], [2, 5], [8, 4], [8, 5]]
        self.reset()

    def is_in_region(self, region, xy_point):
        """
        Check if a given point is within a specified region.

        Args:
            region: A list of coordinates defining the region.
            xy_point: The point to check (x, y).

        Returns:
            bool: True if the point is in the region, False otherwise.
        """
        # Check if the random coordinate is within the rectangle
        x, y = xy_point
        x_min, y_min = min(point[0] for point in region), min(point[1] for point in region)
        x_max, y_max = max(point[0] for point in region), max(point[1] for point in region)

        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False

    def move(self, cur_position, angle):
        """
        Move the agent's position based on the specified angle.

        Args:
            cur_position: Current position of the agent (x, y).
            angle: The angle in degrees divided by 10

        Returns:
            list: The new position of the agent (x, y).
        """
        # Current position
        x, y = cur_position[0], cur_position[1]

        # Convert degrees to radians
        angle_in_rad = math.radians((angle+1)*10)

        # Calculate the change in x and y coordinates based on the current angle
        delta_x = math.cos(angle_in_rad)
        delta_y = math.sin(angle_in_rad)

        # Update the agent's position
        x += delta_x
        y += delta_y

        # Clip new position to defined boundaries
        x = np.clip(x, 0, self.max_row)
        y = np.clip(y, 0, self.max_col)

        return [round(x, 1), round(y, 1)]

    def reset(self, random=False):
        """
        Reset the environment to its initial state.

        Args:
            random (bool, optional): If True, reset to a random state.

        Returns:
            list: The initial or random state (x, y).
        """
        # Use exploring starts by default
        self.is_done = False
        if random:
            while True:
                self.cur_state = [np.random.randint(self.max_row + 1), np.random.randint(self.max_col + 1)]
                if self.cur_state not in [self.terminal]:
                    break
        else:
            self.cur_state = deepcopy(self.initial_pos)
        return self.cur_state

    def _get_state_dim(self):
        """
        Get the dimensions of the state space.

        Returns:
            list: The dimensions of the state space
        """
        return [10, 10]

    def _get_action_dim(self):
        """
        Get the dimensionality of the action space.

        Returns:
            int: The number of possible actions, which is 36.
        """
        return 36 # 36 possible actions (10 degrees per action)

    def transition(self, state, action):
        """
        Perform a transition based on the given state and action.

        Args:
            state: The current state (x, y).
            action: The action to take.

        Returns:
            tuple: A tuple containing the reward, next state, and a flag indicating if the episode is done.
        """
        if self.is_done:
            return 0, state, True

        current_state = deepcopy(state)
        next_state = self.move(current_state, action)

        if self.is_in_region(self.goal, next_state):
            reward = 100   # goal
            if self.verbose: print(f"Reached goal, rewards: {reward}, state: {next_state}")
            self.is_done = True
        elif self.is_in_region(self.trap, next_state):
            reward = -50   # trap
            if self.verbose: print(f"stepped on trap, rewards: {reward}, state: {next_state}")
        else:
            reward = -1    # normal step
            self.is_done = False
            if self.verbose: print(f"rewards: {reward}, state: {next_state}")

        return reward, next_state, self.is_done