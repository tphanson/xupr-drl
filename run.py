import os
import tensorflow as tf

from env import OhmniInSpace
from agent.dqn import DQN

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Saving dir
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
env = OhmniInSpace.env(gui=True)

# Agent
dqn = DQN(env, CHECKPOINT_DIR)
dqn.load_checkpoint()

counter = 0
promote_step = 100000
step = dqn.agent.train_step_counter.numpy()
state = dqn.q_net.get_initial_state()
# difficulty = min(step // promote_step, 15)
difficulty = 0
OhmniInSpace.promote_difficulty(env, difficulty)
while counter < 10000:
    counter += 1
    time_step = env.current_time_step()
    policy_step = dqn.agent.policy.action(time_step, state)
    action, state, _ = policy_step
    if time_step.is_last():
        state = dqn.q_net.get_initial_state()
    _, reward, _, _ = env.step(action)
    print('Action: {}, Reward: {}'.format(action.numpy(), reward.numpy()))
    env.render()
