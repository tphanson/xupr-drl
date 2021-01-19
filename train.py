import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
# import tensorflow as tf
from tf_agents.policies import random_tf_policy

from env import OhmniInSpace
from agent import network
from buffer import reb
# from criterion import ExpectedReturn

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: training servers
# LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0


# Environment
train_env = OhmniInSpace.env()

# Agent
agent = network.Network(
    time_step_spec=train_env.time_step_spec(),
    observation_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
)

# Metrics and Evaluation
# ER = ExpectedReturn()

# Replay buffer
initial_collect_steps = 32  # 2000
replay_buffer = reb.ReplayExperienceBuffer(
    agent.data_spec,
    batch_size=train_env.batch_size
)
# Init buffer
random_policy = random_tf_policy.RandomTFPolicy(
    time_step_spec=agent.time_step_spec,
    action_spec=agent.action_spec,
)
replay_buffer.collect_steps(
    train_env, random_policy,
    steps=initial_collect_steps
)
dataset = replay_buffer.as_dataset(n_steps=agent.get_n_steps())
iterator = iter(dataset)

# Train
num_iterations = 200000
# eval_step = 1000
start = time.time()
loss = 0
while agent.get_step() <= num_iterations:
    replay_buffer.collect_steps(train_env, agent)
    experience, _ = next(iterator)
    loss += agent.train(experience)
    # if agent.get_step() % eval_step == 0:
    #     # Evaluation
    #     avg_return = ER.eval()
    #     print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
    #         agent.get_step(), avg_return, loss / eval_step))
    #     end = time.time()
    #     print('Step estimated time: {:.4f}'.format((end - start) / eval_step))
    #     # Reset
    #     start = time.time()
    #     loss = 0
