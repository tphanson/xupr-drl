from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ReplayExperienceBuffer:
    def __init__(self, data_spec, batch_size=1, n_steps=2):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.replay_buffer_capacity = 10000
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.replay_buffer_capacity,
        )
        self.state = None

    def __len__(self):
        return self.buffer.num_frames()

    def _add_batch(self, traj):
        self.buffer.add_batch(traj)

    def clear(self):
        return self.buffer.clear()

    def as_dataset(self, sample_batch_size=32):
        return self.buffer.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=self.n_steps
        ).prefetch(3)

    def collect(self, env, policy):
        if self.state is None:
            self.state = policy.get_initial_state(batch_size=1)
        time_step = env.current_time_step()
        policy_step = policy.action(time_step, self.state)
        action, self.state, _ = policy_step
        next_time_step = env.step(action)
        traj = trajectory.from_transition(
            time_step, policy_step, next_time_step)
        self._add_batch(traj)
        if traj.is_last():
            self.state = policy.get_initial_state(batch_size=1)
        return traj

    def collect_steps(self, env, policy, steps=1):
        for _ in range(steps):
            self.collect(env, policy)
