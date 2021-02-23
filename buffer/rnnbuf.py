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
        yield step_types, start_state, start_policy_state, action, rewards, end_state,  end_policy_state

    def pipeline(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            args=[],
            output_types=(tf.int32, tf.float32, tf.float32,
                          tf.int32, tf.float32, tf.float32, tf.float32),
            output_shapes=((self.batch_size,), (self.batch_size, 96, 96, 3), (self.batch_size, self.rnn_units, self.rnn_units),
                           (self.batch_size,), (self.batch_size,), (self.batch_size, 96, 96, 3), (self.batch_size, self.rnn_units, self.rnn_units))
        )
