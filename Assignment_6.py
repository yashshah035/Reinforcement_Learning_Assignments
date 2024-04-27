import numpy as np

class Sarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.Q = np.zeros((n_states, n_actions))  # Q table

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)  # Random action
        else:
            return np.argmax(self.Q[state, :])  # Greedy action

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q[next_state, next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

def main():
    # Environment parameters
    n_states = 10
    n_actions = 2

    # SARSA parameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 1000

    # Initialize SARSA agent
    agent = Sarsa(n_states, n_actions, alpha, gamma, epsilon)

    for episode in range(episodes):
        state = np.random.randint(0, n_states)  # Initial state
        action = agent.choose_action(state)  # Initial action

        while True:
            # Take action and observe next state and reward
            next_state = state + np.random.choice([-1, 1])  # Move left or right
            if next_state < 0:
                next_state = 0
            elif next_state >= n_states:
                next_state = n_states - 1
            reward = 1 if next_state == n_states - 1 else 0  # Reward is 1 at goal state

            # Choose next action
            next_action = agent.choose_action(next_state)

            # Update Q values
            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            if state == n_states - 1:
                break

    # Print learned Q values
    print("Learned Q values:")
    print(agent.Q)

if __name__ == "__main__":
    main()
