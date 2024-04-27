import gym 
import numpy as np
import imageio

env = gym.make('CartPole-v1')

num_episodes = 10

# Define your policy class or function here
class RandomPolicy:
    def act(self, state):
        return env.action_space.sample(), None  # Return a random action

# Function to record video
def record_video(env, policy, out_directory, fps=30):
    images = []
    done = False
    state = env.reset()
    img = env.render(mode="rgb_array")
    images.append(img)
    while not done:
        action, _ = policy.act(state)
        state, reward, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

for episode in range(num_episodes):
    state = env.reset()
    state = state  # No need to access state[0] for CartPole
    
    total_reward = 0
    done = False
    
    # Run the episode until it terminates
    while not done:
        env.render()
        
        action = env.action_space.sample()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        state = next_state
    
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

    # Record video after each episode
    record_video(env, RandomPolicy(), f"episode_{episode+1}.mp4")

env.close()
