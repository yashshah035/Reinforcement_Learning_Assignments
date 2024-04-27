import random

class GridWorld:
    def __init__(self, rows, cols, start, goal, obstacles):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        reward = -1  # Default reward for each step

        if action == "up":
            new_row = max(row - 1, 0)
            new_col = col
        elif action == "down":
            new_row = min(row + 1, self.rows - 1)
            new_col = col
        elif action == "left":
            new_row = row
            new_col = max(col - 1, 0)
        else:  # action == "right"
            new_row = row
            new_col = min(col + 1, self.cols - 1)

        new_state = (new_row, new_col)

        if new_state == self.goal:
            reward = 0
            done = True
        elif new_state in self.obstacles:
            new_state = self.state  # Stay in the same state
            reward = -10  # Penalty for hitting an obstacle
            done = False
        else:
            done = False

        self.state = new_state
        return self.state, reward, done, None

class MonteCarloValueEstimation:
    def __init__(self, env, gamma=0.9, alpha=0.1, num_episodes=1000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.state_values = {}
        self.state_visit_counts = {}

    def get_value(self, state):
        return self.state_values.get(state, 0.0)

    def update_value(self, state, value):
        old_value = self.get_value(state)
        visit_count = self.state_visit_counts.get(state, 0)
        new_value = old_value + self.alpha * (value - old_value) / (visit_count + 1)
        self.state_values[state] = new_value
        self.state_visit_counts[state] = visit_count + 1

    def estimate_value_function(self, policy):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_states = [state]
            episode_rewards = []

            done = False
            while not done:
                action = policy(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_states.append(next_state)
                episode_rewards.append(reward)
                state = next_state

            episode_returns = []
            G = 0
            for reward in reversed(episode_rewards):
                G = self.gamma * G + reward
                episode_returns.insert(0, G)

            for state, G in zip(episode_states, episode_returns):
                self.update_value(state, G)

    def get_value_function(self):
        return self.state_values

def random_policy(state):
    actions = ["up", "down", "left", "right"]
    return random.choice(actions)

if __name__ == "__main__":
    # Define a simple 4x4 Grid World
    rows, cols = 4, 4
    start = (0, 0)
    goal = (3, 3)
    obstacles = [(1, 1), (2, 2)]

    env = GridWorld(rows, cols, start, goal, obstacles)
    mc = MonteCarloValueEstimation(env)
    mc.estimate_value_function(random_policy)
    value_function = mc.get_value_function()

    for state, value in value_function.items():
        print(f"State: {state}, Value: {value:.2f}")
