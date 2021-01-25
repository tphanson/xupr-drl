import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import tensorflow as tf
from tf_agents.policies import random_tf_policy

from env import OhmniInSpace
from agent import network
from buffer import per
from criterion import ExpectedReturn

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
ER = ExpectedReturn()

# Replay buffer
initial_collect_steps = 2000
replay_buffer = per.PrioritizedExperienceRelay(
    agent.data_spec,
    n_steps=agent.get_n_steps(),
    batch_size=train_env.batch_size
)
# Init buffer
random_policy = random_tf_policy.RandomTFPolicy(
    time_step_spec=agent.time_step_spec,
    action_spec=agent.action_spec,
    policy_state_spec=agent.policy_state_spec,
)
replay_buffer.collect_steps(
    train_env, random_policy,
    steps=initial_collect_steps
)
dataset = replay_buffer.as_dataset()
iterator = iter(dataset)

# Train
num_iterations = 2000000
eval_step = agent.get_callback_period()
start = time.time()
loss = 0
while agent.get_step() <= num_iterations:
    replay_buffer.collect_steps(train_env, agent)
    experience, info = next(iterator)
    key, probability, table_size, priority = info

    # Debug
    _priority, _ = tf.split(
        priority,
        num_or_size_splits=[1, agent.get_n_steps() - 1],
        axis=-1
    )
    _priority = tf.squeeze(_priority)
    print(_priority)

    mean_loss, batch_loss = agent.train(experience)
    new_priority = tf.multiply(
        tf.ones(priority.shape, dtype=tf.float32),
        tf.expand_dims(batch_loss / agent.get_n_steps(), axis=-1))
    key = tf.reshape(key, shape=[-1]).numpy()
    new_priority = tf.reshape(new_priority, shape=[-1]).numpy()
    updates = {}
    for _key, _new_priority in zip(key, new_priority):
        updates[_key] = _new_priority
    replay_buffer.update_priority(updates)
    loss += mean_loss
    if agent.get_step() % eval_step == 0:
        # Evaluation
        avg_return = ER.eval()
        print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
            agent.get_step(), avg_return, loss / eval_step))
        end = time.time()
        print('Step estimated time: {:.4f}'.format((end - start) / eval_step))
        # Reset
        start = time.time()
        loss = 0
        ER.save()
