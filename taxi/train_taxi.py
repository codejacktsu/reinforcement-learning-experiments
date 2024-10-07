import gymnasium as gym
import numpy as np
import pickle


# Setup Env
env = gym.make("Taxi-v3", render_mode="rgb_array")
env.reset()


def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

# policies
def greedy_policy(q_table, state):
    return np.argmax(q_table[state, :])

def epsilon_greedy_policy(q_table, state, epsilon):
    random_num = np.random.random()
    if random_num > epsilon:
        action = greedy_policy(q_table, state)
    else:
        action = env.action_space.sample()
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, learning_rate, gamma, max_steps, Qtable):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)
            Qtable[state, action] = Qtable[state, action] + learning_rate * (reward + gamma * np.max(Qtable[observation, :]) - Qtable[state, action])
            if terminated or truncated:
                break
            state = observation
    return Qtable

state_space = env.observation_space.n
action_space = env.action_space.n

q_table = initialize_q_table(state_space, action_space)

# Hyperparameters
# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability
epsilon_decay = 0.005            # Exponential decay rate for exploration prob

q_table = train(n_training_episodes, min_epsilon, max_epsilon, epsilon_decay, env, learning_rate, gamma, max_steps, q_table)

# Save Model
with open("qlearning-taxi.pkl", "wb") as f:
    pickle.dump(q_table, f)

# Evaluate Model
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table, None)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
