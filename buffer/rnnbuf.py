import tensorflow as tf

from helper.utils import parse_experiences


class RNNBuffer:
    def __init__(self, replay_buffer, hidden_states_fn, pre_n_steps, n_steps, rnn_units, batch_size=32):
        self.replay_buffer = replay_buffer
        self.hidden_states_fn = hidden_states_fn
        self.pre_n_steps = pre_n_steps
        self.n_steps = n_steps
        self.rnn_units = rnn_units
        self.batch_size = batch_size

        self.iterator = iter(self.replay_buffer.as_dataset())

    def generator(self):
        experiences, info = next(self.iterator)
        start_policy_state, end_policy_state = self.hidden_states_fn(
            experiences)
        step_types, start_state, action, rewards, end_state = parse_experiences(
            experiences, self.pre_n_steps, self.n_steps)
        print(1, step_types.shape)
        print(2, start_state.shape)
        print(3, start_policy_state[0].shape, start_policy_state[1].shape)
        print(4, action.shape)
        print(5, rewards.shape)
        print(6, end_state.shape)
        print(7, end_policy_state[0].shape, end_policy_state[1].shape)
        yield step_types, start_state, start_policy_state, action, rewards, end_state, end_policy_state

    def pipeline(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            args=[],
            output_types=(tf.int32, tf.float32, tf.float32,
                          tf.int32, tf.float32, tf.float32, tf.float32),
            output_shapes=(
                (self.batch_size, self.n_steps),
                (self.batch_size, 96, 96, 3),
                (2, self.batch_size, self.rnn_units),
                (self.batch_size,),
                (self.batch_size, self.n_steps-1),
                (self.batch_size, 96, 96, 3),
                (2, self.batch_size, self.rnn_units)
            )
        )
