import gymnasium as gym
from stable_baselines3 import DQN


game = "ALE/SpaceInvaders-v5"
env = gym.make(game, render_mode="human")

model = DQN.load("spaceinvaders_dqn_model")

episodes = 5
for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += rewards
        steps += 1
        env.render()
    
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}")

env.close()
