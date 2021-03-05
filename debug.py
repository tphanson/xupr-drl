import numpy as np

gamma = 0.99
positive_reward = 10
negative_reward = -10

for i in range(500):
    positive_reward = gamma * positive_reward - 0
for i in range(500):
    negative_reward = gamma * negative_reward - 0

print('positive_reward', positive_reward)
print('negative_reward', negative_reward)

print(np.linspace(-10, 10, 51))
