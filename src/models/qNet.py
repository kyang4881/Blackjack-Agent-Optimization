import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

# Define the neural network for Q-value approximation
class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Define fully connected layers for the neural network
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_dim)

    def forward(self, x):
        """
        Forward pass of the QNetwork.

        Args:
            x: Input state.

        Returns:
            x: Q-values for the given state.
        """
        # Perform a series of fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class TemporalDifference:
    """
    Temporal Difference learning agent.

    Args:
        Env: The environment.
        alpha (float, optional): Learning rate.
        gamma (float, optional): Discount factor.
        epsilon (float, optional): Exploration rate.
        lambd (float, optional): Lambda parameter for eligibility traces.
        verbose (bool, optional): Whether to print verbose information.
    """
    def __init__(self, Env, alpha=0.1, gamma=0.9, epsilon=0.1, lambd=0.9, verbose=False):
        self.Env = Env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        # Get the dimensionality of the state and action space
        self.state_dim = self.Env._get_state_dim()
        self.action_dim = self.Env._get_action_dim()
        # Create the Q-network for Q-value approximation
        self.Q_network = QNetwork(len(self.state_dim), self.action_dim)
        # Optimizer for updating weights
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.alpha)
        # Loss function for backpropagation
        self.loss_fn = nn.MSELoss()
        # Initialize eligibility traces
        self.E = torch.zeros(1, self.action_dim)

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        Args:
            state: The current state.

        Returns:
            action: The selected action.
        """
        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Get Q-values from the Q-network
        q_values = self.Q_network(state_tensor)
         # Choose an action with epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = q_values.argmax().item()
        return action

    def train(self, num_episodes, on_policy=True, loop_iter_limit=5000):
        """
        Train the agent using temporal difference learning with a function approximator network.

        Args:
            num_episodes (int): The number of episodes to train.
            on_policy (bool, optional): Whether to use on-policy learning.
            loop_iter_limit (int): End the while loop after the specified number of iterations
        """

        for _ in tqdm(range(num_episodes)):
            # Reset the environment and choose an initial action
            state = self.Env.reset()
            action = self.choose_action(state)
            rewards = []
            iter = 0

            if self.lambd == 1:
                # Initialize episode memory for lambd = 1
                episode_memory = []

            while not self.Env.is_done:
                # Take a step in the environment and observe the next state and reward
                reward, next_state, done = self.Env.transition(state, action)
                # Choose the next action based on the next state
                next_action = self.choose_action(next_state)
                rewards.append(reward)

                # Teriminate the loop if the tthreshold is reached
                if iter > loop_iter_limit:
                    #print(f"Terminating after {loop_iter_limit} iterations.")
                    break
                else:
                    iter += 1

                if self.lambd == 1:
                    # Store state, action, and reward
                    episode_memory.append((state, action, reward))
                    state, action = next_state, next_action
                    continue

                if on_policy:
                    # Compute the temporal difference error for on-policy learning
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    q_values = self.Q_network(state_tensor)
                    next_q_values = self.Q_network(next_state_tensor)
                    delta = reward + self.gamma * (next_q_values[next_action] - q_values[action])

                else:
                    # Compute the temporal difference error for off-policy learning
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    q_values = self.Q_network(state_tensor)
                    next_q_values = self.Q_network(next_state_tensor)
                    best_next_action = next_q_values.argmax()
                    delta = reward + self.gamma * (next_q_values[best_next_action] - q_values[action])

                # Update eligibility traces
                self.E *= self.gamma * self.lambd
                self.E[0, action] += 1
                # Perform backpropagation and update the Q-network
                self.optimizer.zero_grad()
                # Prediction value
                pred_val = self.Q_network.forward(torch.tensor(state, dtype=torch.float32))
                # True value
                true_val = torch.tensor(self.alpha * delta * self.E).detach().numpy()
                # Compute loss, propagate gradients, and update network
                loss = self.loss_fn(torch.tensor(pred_val, requires_grad=True), torch.tensor(true_val, requires_grad=True))
                loss.backward()
                self.optimizer.step()
                # Transition to the next state and choose the next action
                state, action = next_state, next_action

            # If lambd is 1, perform retrospective updates using episode memory
            if self.lambd == 1:
                G = 0
                for state, action, reward in reversed(episode_memory):
                    G = reward + self.gamma * G
                    state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True)
                    q_values = self.Q_network(state_tensor)
                    delta = G - q_values[action]
                    self.optimizer.zero_grad()
                    # Prediction value
                    pred_val = self.Q_network.forward(torch.tensor(state, dtype=torch.float32))
                    # True value
                    true_val = torch.tensor(self.alpha * delta * self.E).detach().numpy()
                    # Compute loss, propagate gradients, and update network
                    loss = self.loss_fn(torch.tensor(pred_val, requires_grad=True), torch.tensor(true_val, requires_grad=True))
                    loss.backward()
                    self.optimizer.step()


def test_agent(agent, env, num_episodes=1, verbose=False, loop_iter_limit=500):
    """
    Test an agent in an environment by running episodes and calculating the average reward.

    Args:
        agent: The agent to be tested.
        num_episodes (int, optional): Number of episodes to run
        loop_iter_limit (int): End the while loop after the specified number of iterations
    """
    rewards = []
    i = 0
    iter = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        if verbose: print("state: ", state)
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            #q_values = agent.Q_network(state_tensor)
            # Choose an action based on the Q values
            #action = q_values.argmax().item()
            action = agent.Q_network.forward(state_tensor).argmax().item()
            reward, next_state, done = env.transition(state, action)
            if verbose and (i % 50==0 or i in np.arange(10)):
                print(f"Iteration: {i}, reward: {reward}, next_state {next_state}, done: {done}, action: {action}")
            i += 1
            episode_reward += reward
            state = next_state

            # Teriminate the loop if the tthreshold is reached
            if iter > loop_iter_limit:
                break
            else:
                iter += 1

        rewards.append(episode_reward)

    avg_reward = sum(rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")