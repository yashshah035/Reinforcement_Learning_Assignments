import numpy as np

class Environment:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def step(self, state, action):
        next_state = state + action
        reward = 0
        if next_state == self.num_states - 1:
            reward = 1  # Reward of 1 for reaching the goal state
        return next_state, reward

class TD_Prediction:
    def __init__(self, num_states, num_actions, n, alpha, gamma, lambda_val):
        self.num_states = num_states
        self.num_actions = num_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.value_function = np.zeros(num_states)
        self.eligibility_trace = np.zeros(num_states)

    def update_value_function(self, episode):
        T = len(episode)
        for t in range(T - 1):
            # Calculate n-step return
            n_step_return = 0
            for i in range(t + 1, min(t + self.n, T)):
                n_step_return += (self.gamma ** (i - t - 1)) * episode[i][1]
            if t + self.n < T:
                n_step_return += (self.gamma ** self.n) * self.value_function[episode[t + self.n][0]]

            # Update eligibility traces
            self.eligibility_trace *= self.gamma * self.lambda_val
            self.eligibility_trace[episode[t][0]] += 1

            # Update value function
            td_error = n_step_return - self.value_function[episode[t][0]]
            self.value_function += self.alpha * td_error * self.eligibility_trace

    def train(self, env, num_episodes):
        for _ in range(num_episodes):
            state = 0  # Starting state
            episode = []
            while True:
                action = np.random.randint(env.num_actions)  # Random policy
                next_state, reward = env.step(state, action)
                episode.append((state, reward))
                if next_state == env.num_states - 1:  # Reached the goal state
                    break
                state = next_state

            self.update_value_function(episode)

if __name__ == "__main__":
    num_states = 10
    num_actions = 2
    n = 3
    alpha = 0.1
    gamma = 0.9
    lambda_val = 0.5
    num_episodes = 1000

    env = Environment(num_states, num_actions)
    td_prediction = TD_Prediction(num_states, num_actions, n, alpha, gamma, lambda_val)
    td_prediction.train(env, num_episodes)

    print("Estimated Value Function:")
    print(td_prediction.value_function)
