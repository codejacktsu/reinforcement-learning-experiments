import pickle
import gymnasium as gym
import numpy as np
from train_frozenlake import greedy_policy


with open("qlearning-frozenlake-4x4-slippery.pkl", "rb") as f:
    q_table = pickle.load(f)

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, max_episode_steps=1000, render_mode="rgb_array")

# Run the model for a few episodes
for episode in range(2):
    # Create a new environment for inference
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")

    # Reset the environment
    obs, _ = env.reset()

    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        # Get the model's action
        action = greedy_policy(q_table, obs)
        
        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
    
    print(f"Episode {episode + 1} reward: {episode_reward}")
    
    # Reset the environment for the next episode
    obs, _ = env.reset()

# Close the environment
env.close()
