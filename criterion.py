import ray
import numpy as np

from env import OhmniInSpace
from agent import network

ray.init()

ONE_GIGABYTES = 1024 * 1024 * 1024


@ray.remote(memory=2 * ONE_GIGABYTES)
class EvalActor(object):
    def __init__(self):
        self.env = OhmniInSpace.env()
        for pyenv in self.env.envs:
            self.max_steps = pyenv.max_steps
            break
        self.agent = network.Network(
            time_step_spec=self.env.time_step_spec(),
            observation_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            training=False
        )

    def eval(self):
        time_step = self.env.reset()
        steps = self.max_steps
        episode_return = 0.0
        state = self.agent.get_initial_state()
        while not time_step.is_last():
            steps -= 1
            policy_step = self.agent.action(time_step, state)
            action, state, _ = policy_step
            time_step = self.env.step(action)
            episode_return += time_step.reward
        episode_return += time_step.reward * steps
        return episode_return.numpy()[0]


class ExpectedReturn:
    def __init__(self):
        self.filename = 'models/eval.npy'
        self.returns = self.load()

    def eval_multiple_episodes(self, num_episodes):
        actors = [
            EvalActor.remote()
            for _ in range(num_episodes)
        ]
        futures = [
            actor.eval.remote()
            for actor in actors
        ]
        episode_returns = ray.get(futures)
        avg_return = sum(episode_returns) / num_episodes
        # Release memory
        for actor in actors:
            ray.kill(actor)
        del futures
        return avg_return

    def eval(self, num_episodes=5):
        avg_return = self.eval_multiple_episodes(num_episodes)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def load(self):
        try:
            data = np.load(self.filename)
            return list(data)
        except IOError:
            return None

    def save(self):
        data = np.array(self.returns)
        np.save(self.filename, data)
