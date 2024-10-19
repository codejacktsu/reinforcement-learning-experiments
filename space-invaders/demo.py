import gymnasium as gym
import torch
import numpy as np
from train_v2 import DQN, preprocess_state  # Import the DQN class and preprocess_state function

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")

# Get the state shape and action size
state_shape = (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
action_size = env.action_space.n

# Initialize the DQN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_shape, action_size).to(device)

# Load the saved model weights
model.load_state_dict(torch.load('space_invaders_dqn_model.pth'))
model.eval()  # Set the model to evaluation mode

obs = env.reset()
# obs = preprocess_state(obs)

while True:
    state = torch.FloatTensor(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values).item()

    obs, reward, terminated, truncated, info = env.step(action)
    # obs = preprocess_state(obs)
    if terminated or truncated:
        obs = env.reset()
        # obs = preprocess_state(obs)
