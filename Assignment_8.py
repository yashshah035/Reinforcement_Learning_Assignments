from collections import deque
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_size: int) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state: List) -> Tuple[int, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy: Policy, optimizer: torch.optim.Optimizer,
              n_training_episodes: int, max_t: int, gamma: float,
              print_every: int
              ) -> List[int]:

    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

    return scores


def evaluate_agent(env: gym.Env, max_steps: int,
                   n_eval_episodes: int, policy: Policy
                   ) -> Tuple:
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state = env.reset()[0]
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    env_id = 'CartPole-v1'
    env = gym.make(env_id)

    eval_env = gym.make(env_id)

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print('Observation Space')
    print('The State Space is: ', s_size)
    print('Sample observation', env.observation_space.sample())

    print('Action Space')
    print('The Action Space is: ', a_size)
    print('Action Space Sample', env.action_space.sample())

    cartpole_hyperparameters = {
        'h_size': 16,
        'n_training_episodes': 1000,
        'n_evaluation_episodes': 10,
        'max_t': 1000,
        'gamma': 1.0,
        'lr': 1e-2,
        'env_id': env_id,
        'state_space': s_size,
        'action_space': a_size,
    }

    cartpole_policy = Policy(cartpole_hyperparameters['state_space'],
                             cartpole_hyperparameters['action_space'],
                             cartpole_hyperparameters['h_size']).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(),
                                    lr=cartpole_hyperparameters['lr'])

    scores = reinforce(cartpole_policy,
                       cartpole_optimizer,
                       cartpole_hyperparameters['n_training_episodes'],
                       cartpole_hyperparameters['max_t'],
                       cartpole_hyperparameters['gamma'],
                       100)

    eval = evaluate_agent(eval_env,
                          cartpole_hyperparameters['max_t'],
                          cartpole_hyperparameters['n_evaluation_episodes'],
                          cartpole_policy)
    print('Evaluation: ', eval)
