import reverb
import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.trajectories import trajectory, time_step, policy_step


class PrioritizedExperienceRelay:
    def __init__(self, data_spec, batch_size=1, n_steps=2):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.replay_buffer_capacity = 100000

        self.name = 'PER'
        self.server = reverb.Server([
            reverb.Table(
                name=self.name,
                max_size=self.replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(0.8),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1)
            )
        ])
        self.buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.data_spec,
            sequence_length=self.n_steps,
            table_name=self.name,
            local_server=self.server
        )
        self.writer = reverb_utils.ReverbAddTrajectoryObserver(
            self.buffer.py_client,
            table_name=self.name,
            sequence_length=self.n_steps,
            stride_length=1,
            priority=1,
        )
        self.states = None

    def __len__(self):
        return self.buffer.num_frames()

    def _add_batch(self, time_steps, policy_steps, next_time_steps):
        for i in range(self.batch_size):
            ts = time_step.TimeStep(
                time_steps.step_type[i],
                time_steps.reward[i],
                time_steps.discount[i],
                time_steps.observation[i],
            )
            ps = policy_step.PolicyStep(
                policy_steps.action[i],
                policy_steps.state[i],
                (),
            )
            nts = time_step.TimeStep(
                next_time_steps.step_type[i],
                next_time_steps.reward[i],
                next_time_steps.discount[i],
                next_time_steps.observation[i],
            )
            traj = trajectory.from_transition(ts, ps, nts)
            self.writer(traj)

    def clear(self):
        return self.buffer.clear()

    def as_dataset(self, sample_batch_size=32):
        return self.buffer.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=self.n_steps
        ).prefetch(3)

    def update_priority(self, updates):
        self.buffer.py_client.mutate_priorities(self.name, updates)

    def collect(self, env, policy):
        if self.states is None:
            self.states = policy.get_initial_state(batch_size=self.batch_size)
        time_steps = env.current_time_step()
        policy_steps = policy.action(time_steps, self.states)
        actions, policy_states, _ = policy_steps
        next_time_steps = env.step(actions)
        self._add_batch(time_steps, policy_steps, next_time_steps)
        # Reset states
        not_lasts = tf.cast(
            tf.less(time_steps.step_type, time_step.StepType.LAST),
            tf.float32
        )
        [hidden_states, carry_states] = policy_states
        hidden_states = tf.multiply(hidden_states, not_lasts)
        carry_states = tf.multiply(carry_states, not_lasts)
        self.states = [hidden_states, carry_states]

    def collect_steps(self, env, policy, steps=1):
        for _ in range(steps):
            self.collect(env, policy)
