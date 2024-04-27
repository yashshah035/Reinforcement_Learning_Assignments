import numpy as np

# Define MDP parameters
num_states = 5
num_actions = 2
gamma = 0.9  # discount factor

# Initialize random MDP
transition_probs = np.random.rand(num_states, num_actions, num_states)
transition_probs /= transition_probs.sum(axis=2, keepdims=True)  # Normalize
rewards = np.random.rand(num_states, num_actions)

# Value iteration
def value_iteration(transition_probs, rewards, gamma, epsilon=1e-6):
    num_states, num_actions, _ = transition_probs.shape
    V = np.zeros(num_states)  # Initialize value function

    while True:
        V_new = np.zeros(num_states)
        for s in range(num_states):
            Q = np.zeros(num_actions)
            for a in range(num_actions):
                Q[a] = np.sum(transition_probs[s, a] * (rewards[s, a] + gamma * V))
            V_new[s] = np.max(Q)
        if np.max(np.abs(V - V_new)) < epsilon:
            break
        V = V_new

    return V

optimal_value_function = value_iteration(transition_probs, rewards, gamma)
print("Optimal Value Function:", optimal_value_function)
