import numpy as np

class MonteCarloAgent:
    """
    Initialize the Monte Carlo Agent.

    Args:
        env: The environment the agent interacts with.
        epsilon: The exploration-exploitation trade-off parameter.
        num_episodes: The number of episodes to run during training.
        verbose: If True, the agent will print episode information.
    """
    def __init__(self, env, epsilon=0.1, gamma=0.9, num_episodes=100000, verbose=False):
        self.env = env
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = {}  # Q-table to store state-action values
        self.N = {}  # Keeps track of the number of visits to each state-action pair
        self.verbose = verbose
        self.gamma = gamma

    def generate_episode(self):
        """
        Generate a single episode using the current policy.

        Returns:
            episode: A list of (state, action) tuples representing the episode.
            reward: The total reward obtained in the episode.
        """
        episode = []
        player_hand, dealer_hand = self.env.initial_state()
        done = False

        while not done:
            state = tuple(player_hand)
            if state not in self.Q:
                self.Q[state] = {'hit': 0, 'stand': 0}  # If the satte is not available, initialize with 0

            if np.random.rand() < self.epsilon:  # Randomly choose an action
                action = np.random.choice(['hit', 'stand'])
            else:  # Choose an action based on the Q value of the state-action pair
                action = 'hit' if self.Q[state]['hit'] > self.Q[state]['stand'] else 'stand'

            episode.append((state, action))  # Save state-action pair into episode list
            player_hand, dealer_hand, reward, done = self.env.step(player_hand, dealer_hand, action)

        return episode, reward

    def train(self):
        """
        Train the agent by running multiple episodes and updating the Q-table.
        """
        for _ in range(self.num_episodes):
            episode, reward = self.generate_episode()
            G = 0
            for state, action in episode:
                G = self.gamma * G + reward
                if state not in self.N:
                    self.N[state] = {'hit': 0, 'stand': 0}  # If the satte is not available, initialize with 0
                self.N[state][action] += 1   # Increase the counter if episode already visited
                alpha = 1 / self.N[state][action]
                self.Q[state][action] += alpha* (G - self.Q[state][action])  # Update Q table by normalizing updates to prioritize less frequently visited states

    def choose_action(self, player_hand):
        """
        Choose the action for a given state based on the current Q-values.

        Args:
            player_hand: The current player's hand.

        Returns:
            action: The selected action ('hit' or 'stand').
        """
        state = tuple(player_hand)
        if state not in self.Q:
            self.Q[state] = {'hit': 0, 'stand': 0}
        return 'hit' if self.Q[state]['hit'] > self.Q[state]['stand'] else 'stand'