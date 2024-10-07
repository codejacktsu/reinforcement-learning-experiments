import pickle
import gymnasium as gym
from train_taxi import greedy_policy


# load model
with open("qlearning-taxi.pkl", "rb") as f:
    q_table = pickle.load(f)

env = gym.make("Taxi-v3", render_mode="human")

# Run the model for a few episodes
for episode in range(2):
    # Create a new environment for inference
    env = gym.make("Taxi-v3", render_mode="human")

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
