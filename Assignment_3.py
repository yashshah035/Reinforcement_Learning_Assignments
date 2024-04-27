import numpy as np

class MultiArmedBandit:
    def __init__(self, num_actions, true_action_values):
        self.num_actions = num_actions
        self.true_action_values = true_action_values
        self.action_values = np.zeros(num_actions)  # Estimates of action values
        self.action_counts = np.zeros(num_actions)  # Counts of how many times each action has been chosen

    def choose_action(self, epsilon):
        if np.random.rand() < epsilon:
            # Explore: Choose a random action
            action = np.random.choice(self.num_actions)
        else:
            # Exploit: Choose the action with the highest estimated value
            action = np.argmax(self.action_values)
        return action

    def update_action_value(self, action, reward):
        # Update the estimate of action value using sample-average method
        self.action_counts[action] += 1
        self.action_values[action] += (1 / self.action_counts[action]) * (reward - self.action_values[action])

def main():
    # Define the number of actions and true action values
    num_actions = 5
    true_action_values = np.random.normal(0, 1, num_actions)  # Normally distributed true action values

    # Initialize the multi-armed bandit problem
    bandit = MultiArmedBandit(num_actions, true_action_values)

    num_episodes = 1000  # Number of episodes (plays)

    epsilon = 0.1  # Exploration rate (epsilon-greedy parameter)

    total_rewards = 0
    for episode in range(num_episodes):
        action = bandit.choose_action(epsilon)
        reward = np.random.normal(true_action_values[action], 1)  # Generate reward with noise
        total_rewards += reward
        bandit.update_action_value(action, reward)

    average_reward = total_rewards / num_episodes
    print("Average reward:", average_reward)
 
if __name__ == "__main__":
    main()

