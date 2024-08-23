import numpy as np
from tqdm import tqdm

class TemporalDifference:
    def __init__(self, Env, alpha=0.1, gamma=0.9, epsilon=0.1, lambd=0.9):
        self.Env = Env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd

        self.state_dim = self.Env._get_state_dim()[0]
        self.action_dim = self.Env._get_action_dim()
        self.V = np.zeros(self.state_dim)
        self.Q = np.zeros((self.state_dim, self.action_dim))
        self.E = np.zeros((*self.state_dim, self.action_dim))

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.Q[state[0], state[1], :])

    def train(self, num_episodes, on_policy=True):
        for _ in tqdm(range(num_episodes)):
            episode_memory = []  # to be used when lambd=1
            self.E *= 0
            state = self.Env.reset()
            action = self.epsilon_greedy_policy(state)

            while not self.Env.is_done:
                reward, next_state, done = self.Env.transition(state, action)
                next_action = self.epsilon_greedy_policy(next_state)

                if self.lambd == 1:
                    episode_memory.append((state, action, reward))
                    state, action = next_state, next_action
                    continue

                if on_policy:
                    delta = reward + self.gamma * (
                        self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action]
                    )
                else:
                    best_next_action = np.argmax(self.Q[next_state[0], next_state[1], :])
                    delta = reward + self.gamma * (
                        self.Q[next_state[0], next_state[1], best_next_action] - self.Q[state[0], state[1], action]
                    )

                self.E[state[0], state[1], action] += 1
                self.Q += self.alpha * delta * self.E
                self.E *= self.gamma * self.lambd

                state, action = next_state, next_action

            if self.lambd == 1:
                G = 0
                for state, action, reward in reversed(episode_memory):
                    G = reward + self.gamma * G
                    self.Q[state[0], state[1], action] += self.alpha * (G - self.Q[state[0], state[1], action])