import os
import matplotlib.pyplot as plt
import ray
from tf_agents.utils import common
from agent.dqn import DQN
from env import OhmniInSpace

ray.init()

ONE_GIGABYTES = 1024 * 1024 * 1024


@ray.remote(memory=2 * ONE_GIGABYTES)
class EvalActor(object):
    def __init__(self, num_of_obstacles):
        self.env = OhmniInSpace.env()
        for pyenv in self.env.envs:
            self.max_steps = pyenv.max_steps
            break
        OhmniInSpace.promote_difficulty(self.env, num_of_obstacles)
        self.checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       './models/checkpoints')
        self.dqn = DQN(self.env, self.checkpoint)
        self.dqn.agent.train = common.function(self.dqn.agent.train)

    def eval(self):
        time_step = self.env.reset()
        steps = self.max_steps
        episode_return = 0.0
        state = self.dqn.q_net.get_initial_state()
        while not time_step.is_last():
            steps -= 1
            policy_step = self.dqn.agent.policy.action(time_step, state)
            action, state, _ = policy_step
            time_step = self.env.step(action)
            episode_return += time_step.reward
        episode_return += time_step.reward * steps
        return episode_return.numpy()[0]


class ExpectedReturn:
    def __init__(self):
        self.returns = None

    def eval_multiple_episodes(self, num_episodes, num_of_obstacles):
        actors = [
            EvalActor.remote(num_of_obstacles)
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

    def eval(self, num_episodes=5, num_of_obstacles=0):
        avg_return = self.eval_multiple_episodes(
            num_episodes, num_of_obstacles)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def save(self):
        plt.plot(self.returns)
        plt.ylabel('Average Return')
        plt.savefig('models/eval.jpg')
