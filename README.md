# Reinforcement Learning Assignments

This repository contains various assignments focused on different aspects of Reinforcement Learning (RL). Each assignment includes a Python program that demonstrates a specific RL concept or algorithm.

## Assignments Overview

Assignment_1. **CartPole Environment Evaluation**
   - Demonstrates evaluative feedback by running episodes in the CartPole environment and printing the total reward accumulated in each episode.

Assignment_2. **Markov Decision Processes (MDP) and Value Iteration**
   - Demonstrates Markov Decision Processes and value functions by performing value iteration to find the optimal value function for a randomly initialized MDP.

Assignment_3. **Epsilon-Greedy Algorithm for Multi-Armed Bandit Problem**
   - Demonstrates the explore-exploit dilemma by implementing an epsilon-greedy algorithm for solving a multi-armed bandit problem. It maintains estimates of action values and uses an exploration rate (epsilon) to balance exploration and exploitation.

Assignment_4. **Upper Confidence Bound Algorithm for Binary Bandit Problem**
   - Simulates a binary bandit problem, where each action has a binary (0 or 1) reward. It uses the upper confidence bound algorithm to estimate action values and make decisions based on the highest estimated value.

Assignment_5. **Policy Evaluation with Iterative Simulation**
   - Estimates the value function for a given policy by iteratively simulating episodes and updating the value function based on the observed returns.

Assignment_6. **SARSA Algorithm**
   - Demonstrates SARSA (State-Action-Reward-State-Action), an on-policy Temporal Difference (TD) learning algorithm.

Assignment_7. **n-step TD Prediction using Eligibility Traces**
   - Demonstrates n-step TD prediction using eligibility traces by estimating the value function by iteratively simulating episodes and updating the value function based on n-step returns.

Assignment_8. **REINFORCE Algorithm for Policy Gradient Methods**
   - Demonstrates the REINFORCE algorithm, a policy gradient method for learning optimal policies in reinforcement learning.

Assignment_9. **Q-Learning for Elevator Control**
   - Simulates an elevator's movement between floors, where the agent uses Q-learning to learn the optimal actions for each floor.

## Getting Started

To run these programs, you need to have Python installed along with the necessary libraries. The main library used for these assignments is `gym` for environment simulations.

### Prerequisites

- Python 3.x
- `gym` library
- `numpy` library
- `matplotlib` library (for visualizations, if needed)
- `imageio` library

### Installation

To install the required libraries, you can use `pip`:

```bash
pip install gym numpy matplotlib imageio
```

### Running

'x' should be replace with the actual number of assignment.

```bash
python Assignment_x.py
```

### Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or create a pull request.
