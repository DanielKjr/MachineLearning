import gymnasium as gym
import numpy as np
import pandas as pd
import random

# game = "FrozenLake-v1"
game = "Taxi-v3"

env = gym.make(game, render_mode="ansi")

# Varibale def
gamma = 0.9
alpha = 0.4
training_Length = 1000

Q_table=np.zeros([env.observation_space.n,env.action_space.n])


# Training mode
observation, info = env.reset()

for i in range(training_Length):
    done = False
    observation, info = env.reset()
    while not done:
        current_state = observation
        action = int(np.argmax(Q_table[current_state,])) 

        if action == 0:
            action = env.action_space.sample()

        observation,reward,terminated,truncated,info = env.step(action)
        next_state = observation

        Td = reward + gamma *Q_table[next_state,np.argmax(Q_table[next_state,])] - Q_table[current_state,action]
        Q_table[current_state,action] =  Q_table[current_state,action] + alpha * Td

        done = terminated or truncated 

#Speciel lavet til Taxi
df = pd.DataFrame(Q_table, columns=['Down', 'Up', 'Right', 'Left', 'Pickup', 'Drop off'])
#Speciel lavet til Frozenlake
# df = pd.DataFrame(Q_table, columns=['Down', 'Up', 'Right', 'Left'])
print(df)

env.close()

# Inference mode
# env = gym.make(game, render_mode="human")
# observation, info = env.reset()

# for i in range(15):
#     done = False
#     observation, info = env.reset()
#     while not done:
#         current_state = observation
#         action = int(np.argmax(Q_table[current_state,]))
#         observation, reward, terminated, truncated, info = env.step(action)

#         done = terminated or truncated

# env.close()
