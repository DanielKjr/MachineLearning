import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3', render_mode='human')


num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.99
epsilon = 0.1

observation, info = env.reset(seed=42)
num_episodes = 1000;
for episode in range(num_episodes):
    # action = env.action_space.sample()
    # observation, reward, terminated, truncated,info = env.step(action)

    # if terminated or truncated:
    #     obs = observation, info = env.reset()
    #     print(_)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[observation])

        # next_state, reward, terminated, truncated, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs = observation, info = env.reset()

        Q[info, action] += alpha * (float(reward) * gamma * np.max(Q[observation]) - Q[observation, action])

        total_reward += reward
        observation = observation

    epsilon *= 0.99
    if(episode +1) % 1000 == 0:
        print(f"Episode {episode +1}, Total Reward: {total_reward}")



env.close()

