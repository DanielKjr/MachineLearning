import gymnasium as gym
import numpy as np
import pandas as pd
import random


game = "Taxi-v3"

env = gym.make(game, render_mode="ansi")

gamma = 0.9
alpha = 0.4
epsilon = 0.1
exploration_rate_threshold = 0.2
training_length = 1000

Q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(training_length):
    done = False
    observation, info = env.reset()

    while not done:
        currentState = observation
        
        if random.random() < epsilon:
            action = env.action_space.sample()
            print("ye")
        else:
            action = int(np.argmax(Q_table[currentState,]))


        observation, reward, terminated, truncated, info = env.step(action)
        nextState = observation

        Td = reward + gamma * Q_table[nextState, np.argmax(Q_table[nextState,])] - Q_table[currentState, action]
        Q_table[currentState, action] = Q_table[currentState, action] + alpha * Td

        done = terminated or truncated
        epsilon *= 0.99


env.close()
df = pd.DataFrame(Q_table, columns=['Down', 'Up', 'Right', 'Left', 'Pickup', 'Drop off'])
print(df)

env = gym.make(game, render_mode="human")

for i in range(15):
    done = False
    observation, info = env.reset()
    while not done:
        currentState = observation
        action = int(np.argmax(Q_table[currentState,]))
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

env.close()