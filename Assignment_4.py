import numpy as np

class BinaryBandit:
    def __init__(self, num_actions, true_rewards):
        self.num_actions = num_actions
        self.true_rewards = true_rewards
        self.action_counts = np.zeros(num_actions)
        self.action_values = np.zeros(num_actions)
        self.total_reward = 0
        self.time_step = 0

    def choose_action(self):
        ucb_values = self.action_values + np.sqrt(2 * np.log(self.time_step + 1) / (self.action_counts + 1e-6))
        return np.argmax(ucb_values)

    def take_action(self, action):
        reward = np.random.binomial(1, self.true_rewards[action])
        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]
        self.total_reward += reward
        self.time_step += 1
        return reward

def main():
    num_actions = 5  # Number of actions
    true_rewards = [0.3, 0.5, 0.7, 0.2, 0.8]  # True rewards for each action

    bandit = BinaryBandit(num_actions, true_rewards)

    num_steps = 10  # Number of steps to run the simulation

    for _ in range(num_steps):
        action = bandit.choose_action()
        reward = bandit.take_action(action)
        print(f"Chose action {action}, got reward {reward}, total reward so far: {bandit.total_reward}")

if __name__ == "__main__":
    main()
