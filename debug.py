gamma = 0.99
positive_reward = 10
negative_reward = -10

for i in range(500):
    positive_reward = gamma * positive_reward - 0.2
for i in range(500):
    negative_reward = gamma * negative_reward - 0.2

print('positive_reward', positive_reward) # -1.999999999999999
print('negative_reward', negative_reward) # -1.999999999999999