import numpy as np

# Define constants
NUM_FLOORS = 5
NUM_ACTIONS = 3  # Actions: {0: stay, 1: go up, 2: go down}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
NUM_EPISODES = 1000

class ElevatorAgent:
    def __init__(self, num_floors, num_actions):
        self.num_floors = num_floors
        self.num_actions = num_actions
        self.q_table = np.zeros((num_floors, num_actions))

    def choose_action(self, current_floor):
        if np.random.uniform(0, 1) < EPSILON:
            # Explore
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploit
            action = np.argmax(self.q_table[current_floor])
        return action

    def learn(self, current_floor, action, reward, next_floor):
        old_value = self.q_table[current_floor, action]
        future_rewards = [self.q_table[next_floor, a] for a in range(self.num_actions)]
        next_max = max(future_rewards)
        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        self.q_table[current_floor, action] = new_value

class ElevatorSimulation:
    def __init__(self, num_floors, agent):
        self.num_floors = num_floors
        self.agent = agent
        self.current_floor = 0

    def move(self, action):
        if action == 1 and self.current_floor < self.num_floors - 1:
            self.current_floor += 1
        elif action == 2 and self.current_floor > 0:
            self.current_floor -= 1

    def run_episode(self):
        total_reward = 0
        while True:
            action = self.agent.choose_action(self.current_floor)
            next_floor = self.current_floor
            if action == 1 and self.current_floor < self.num_floors - 1:
                next_floor += 1
            elif action == 2 and self.current_floor > 0:
                next_floor -= 1

            # Reward function
            reward = -1  # Penalty for each move
            if next_floor == self.num_floors - 1:
                reward = 10  # Reward for reaching the top floor

            self.agent.learn(self.current_floor, action, reward, next_floor)
            total_reward += reward
            self.current_floor = next_floor

            if next_floor == self.num_floors - 1:
                break

        return total_reward

# Initialize agent and simulation
agent = ElevatorAgent(NUM_FLOORS, NUM_ACTIONS)
simulation = ElevatorSimulation(NUM_FLOORS, agent)

# Train the agent
for episode in range(NUM_EPISODES):
    total_reward = simulation.run_episode()
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Print the learned Q-table
print("\nLearned Q-table:")
print(agent.q_table)
