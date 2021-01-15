import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy

from env import OhmniInSpace
from agent.dqn import DQN
from buffer import ReplayBuffer
from criterion import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: training servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0

# Saving dir
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
train_env = OhmniInSpace.env()

# Agent
dqn = DQN(train_env, CHECKPOINT_DIR)
dqn.agent.train = common.function(dqn.agent.train)
step = dqn.agent.train_step_counter.numpy()

# Metrics and Evaluation
ER = ExpectedReturn()

# Replay buffer
initial_collect_steps = 2000
replay_buffer = ReplayBuffer(dqn, batch_size=train_env.batch_size)
# Init buffer
random_policy = random_tf_policy.RandomTFPolicy(
    time_step_spec=dqn.agent.collect_policy.time_step_spec,
    action_spec=dqn.agent.collect_policy.action_spec,
    policy_state_spec=dqn.agent.collect_policy.policy_state_spec
)
replay_buffer.collect_steps(
    train_env, random_policy,
    steps=initial_collect_steps
)
dataset = replay_buffer.as_dataset()
iterator = iter(dataset)

# Train
num_iterations = 2000000
eval_step = 1000
promote_step = 100000
start = time.time()
loss = 0
while step <= num_iterations:
    if LOCAL:
        train_env.render()
    replay_buffer.collect_steps(train_env, dqn.agent.collect_policy)
    experience, _ = next(iterator)
    loss += dqn.agent.train(experience).loss
    # Evaluation
    step = dqn.agent.train_step_counter.numpy()
    # difficulty = min(step // promote_step, 15)
    difficulty = 0
    if step % promote_step == 0:
        OhmniInSpace.promote_difficulty(train_env, difficulty)
    if step % eval_step == 0:
        # Checkpoints
        dqn.save_checkpoint()
        # Evaluation
        avg_return = ER.eval(num_of_obstacles=difficulty)
        print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
            step, avg_return, loss / eval_step))
        end = time.time()
        print('Step estimated time: {:.4f}'.format((end - start) / eval_step))
        # Reset
        start = time.time()
        loss = 0

# Visualization
ER.save()
