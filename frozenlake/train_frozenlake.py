import gymnasium as gym
import numpy as np
import pickle


env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, max_episode_steps=1000, render_mode="rgb_array")
env.reset()

state_space = env.observation_space.n
action_space = env.action_space.n

def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

q_table = initialize_q_table(state_space, action_space)

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

# hyperparameters
n_training_episodes = 100000
learning_rate = 0.7

n_eval_episodes = 100

env_id = "FrozenLake-v1"
max_steps = 999
gamma = 0.95
eval_seed = []

max_epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.00005

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        print(epsilon)
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

q_table = train(n_training_episodes, min_epsilon, max_epsilon, epsilon_decay, env, max_steps, q_table)

# save model
policy_name = "qlearning-frozenlake-4x4-slippery.pkl"
with open(policy_name, "wb") as f:
    pickle.dump(q_table, f)

# evaluate model
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
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
