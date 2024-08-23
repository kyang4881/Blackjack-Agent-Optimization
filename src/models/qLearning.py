import random

class QLearningAgent:
    """
    Initialize the Q-learning agent.

    Args:
        env (BlackjackEnvironment): The environment in which the agent operates.
        alpha (float): Learning rate for Q-learning.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Epsilon-greedy exploration parameter.
        num_episodes (int): The number of episodes for training.
        Q (dict): A dictionary containing q values for the states
        verbose (bool): If True, print additional information during training and testing.
    """
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=100000, verbose=False):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy exploration parameter
        self.num_episodes = num_episodes
        self.Q = {}
        self.verbose = verbose

    def choose_action(self, player_hand):
        """
        Choose an action (hit or stand) based on the Q-values for the given player hand.

        Args:
            player_hand (list of int): Player's current hand.

        Returns:
            str: The selected action, either 'hit' or 'stand'.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.player_action(player_hand)  # Explore (random action)
        else:
            # Exploit (choose action with highest Q-value)
            state = tuple(player_hand)
            if state not in self.Q:
                self.Q[state] = {'hit': 0, 'stand': 0}

            hit_value = self.Q[state]['hit']
            stand_value = self.Q[state]['stand']

            if self.verbose and count / 2**power == 1:
                print(f"Q_values:\n---Hit: {float(hit_value)}\n---Stand: {float(stand_value)}")
            if hit_value > stand_value:
                return 'hit'
            elif hit_value == stand_value:
                # Randomly choose between 'hit' and 'stand' when they are the same
                return random.choice(['hit', 'stand'])
            else:
                return 'stand'

    def learn(self, player_hand, player_action, reward, player_hand_next):
        """
        Update the Q-value for a given state-action pair based on the observed reward and next state.

        Args:
            player_hand (list of int): Player's current hand.
            player_action (str): The action taken by the player, either 'hit' or 'stand'.
            reward (float): The reward obtained for taking the specified action.
            player_hand_next (list of int): Player's hand in the next state.
        """
        # Update Q-value using Q-learning formula
        state = tuple(player_hand)
        if state not in self.Q:
            self.Q[state] = {'hit': 0, 'stand': 0}
        # Current Q value
        Q_current = self.Q[state][player_action]
        state_next = tuple(player_hand_next)
        # Optimal future Q value
        Q_future = max(self.Q[state_next]['hit'], self.Q[state_next]['stand'])
        updated_value = (1 - self.alpha) * Q_current + self.alpha * (reward + self.gamma * Q_future)
        # Update Q table
        self.Q[state][player_action] = updated_value

    def train(self):
        """
        Train the Q-learning agent by simulating blackjack games and updating Q-values based on the results.
        """
        global count
        global power
        count = 0
        power = 0
        for episode in range(self.num_episodes):
            player_hand, dealer_hand = self.env.initial_state()
            done = False
            if self.verbose and count / 2**power == 1:
                print(f"Episode: {episode}")
            while not done:
                player_action = self.choose_action(player_hand)
                player_hand_next, dealer_hand_next, reward, done = self.env.step(player_hand, dealer_hand, player_action)
                self.learn(player_hand, player_action, reward, player_hand_next)
                player_hand, dealer_hand = player_hand_next, dealer_hand_next
                #if self.verbose and count / 2**power == 1 and done:
                    #power += 1
            #count += 1