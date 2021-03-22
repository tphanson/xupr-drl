import tensorflow as tf

from env import OhmniInSpace
from agent import network

# Config GPU
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# Environment
env = OhmniInSpace.env(gui=True)

# Agent
agent = network.Network(
    time_step_spec=env.time_step_spec(),
    observation_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    training=False
)

state = agent.get_initial_state()
while True:
    time_step = env.current_time_step()
    policy_step = agent.action(time_step, state)
    action, state, _ = policy_step
    _, reward, _, _ = env.step(action)
    if time_step.is_last():
        state = agent.get_initial_state()

    # Debug
    print('Action: {}, Reward: {}'.format(action.numpy(), reward.numpy()))
    env.render()
