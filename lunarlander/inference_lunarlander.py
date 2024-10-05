# Create a new environment for inference
import gymnasium as gym
from stable_baselines3 import PPO

# Load the trained model
model_name = "ppo-LunarLander-v2"
model = PPO.load(model_name)

env = gym.make("LunarLander-v2", render_mode="human")

# Reset the environment
obs, _ = env.reset()

# Run the model for a few episodes
for episode in range(5):
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        # Get the model's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
    
    print(f"Episode {episode + 1} reward: {episode_reward}")
    
    # Reset the environment for the next episode
    obs, _ = env.reset()

# Close the environment
env.close()
