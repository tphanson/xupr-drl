from env import OhmniInSpace
from agent import network

# Environment
env = OhmniInSpace.env(gui=True)

# Agent
agent = network.Network(
    time_step_spec=env.time_step_spec(),
    observation_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    training=False
)

while True:
    time_step = env.current_time_step()
    policy_step = agent.action(time_step)
    action, _, _ = policy_step
    _, reward, _, _ = env.step(action)

    # Debug
    print('Action: {}, Reward: {}'.format(action.numpy(), reward.numpy()))
    env.render()
