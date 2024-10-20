import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


env_name = "ALE/SpaceInvaders-v5"
env = gym.make(env_name, render_mode="rgb_array")
env = DummyVecEnv([lambda: env])

# Set up the model
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    tensorboard_log="./spaceinvaders_dqn_tensorboard/"
)

# Train the model
total_timesteps = 1000000  # Adjust this based on your needs and computational resources
model.learn(total_timesteps=total_timesteps, log_interval=100)

# Save the trained model
model.save("spaceinvaders_dqn_model")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()