from copy import deepcopy
import numpy as np

class SlipperyWindyCliffGridWorld:
    def __init__(self, grid=None, wind=None, slip_prob=None):
        self.grid = grid or [
            ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
            ['.', '.', '.', '.', '.', '.', 'C'],
            ['.', 'G', '.', '.', '.', '.', 'C'],
            ['.', 'T', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', 'T', '.'],
            ['.', '.', '.', '.', '.', 'S', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['C', 'C', 'C', 'C', 'C', 'C', 'C']
        ]
        self.max_row = len(self.grid) - 1
        self.max_col = len(self.grid[0]) - 1
        self.initial_pos = next(
            [row_index, col_index] for row_index, row in enumerate(self.grid) \
            for col_index, cell in enumerate(row) if cell == 'S'
        )
        self.goal = next(
            [row_index, col_index] for row_index, row in enumerate(self.grid) \
            for col_index, cell in enumerate(row) if cell == 'G'
        )
        self.terminal = list(
            [row_index, col_index] for row_index, row in enumerate(self.grid) \
            for col_index, cell in enumerate(row) if (cell == 'G') or (cell == 'C')
        )
        if wind is None:
            self.wind = [0 for _ in range(self.max_col+1)]    # All zeros
        else:
            self.wind = [i%2 for i in range(self.max_col+1)]  # Wind strength at each column

        self.slip_prob = slip_prob or 0.1  # Probability of slipping (not taking intended action)
        self.reset()

    def reset(self, random=True):
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
        return np.array(self.grid).shape

    def _get_action_dim(self):
        return np.array([4])

    def transition(self, state, action):
        if self.is_done:
            return 0, state, True

        next_state = deepcopy(state)

        # Slippery part
        if np.random.rand() < self.slip_prob:
            action = np.random.choice(4)

        if action == 0:
            next_state[1] += 1   # right
        elif action == 1:
            next_state[0] += 1   # down
        elif action == 2:
            next_state[1] -= 1   # left
        elif action == 3:
            next_state[0] -= 1   # up
        next_state = np.clip(next_state, [0, 0], [self.max_row, self.max_col]).tolist()

        # Windy part
        next_state[0] -= self.wind[next_state[1]]
        next_state[0] = max(0, next_state[0])

        row, col = next_state
        if self.grid[row][col] == 'G':
            reward = 100   # goal
            self.is_done = True
        elif self.grid[row][col] == 'C':
            reward = -200  # cliff
            self.is_done = True
        elif self.grid[row][col] == 'T':
            reward = -50   # trap
        else:
            reward = -1
            self.is_done = False

        return reward, next_state, self.is_done