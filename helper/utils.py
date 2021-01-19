import tensorflow as tf


def parse_experiences(experiences):
    step_types, _ = tf.squeeze(
        tf.split(
            experiences.step_type,
            num_or_size_splits=[1, 1],
            axis=1
        )
    )
    states, next_states = tf.squeeze(
        tf.split(
            experiences.observation,
            num_or_size_splits=[1, 1],
            axis=1
        )
    )
    actions, _ = tf.split(
        experiences.action,
        num_or_size_splits=[1, 1],
        axis=1
    )
    rewards, _ = tf.squeeze(
        tf.split(
            experiences.reward,
            num_or_size_splits=[1, 1],
            axis=1
        )
    )
    return step_types, states, actions, rewards, next_states


def build_mask(batch_size, num_of_atoms):
    return tf.constant([
        [
            [
                i for _ in range(num_of_atoms)
            ] for i in range(num_of_atoms)
        ] for _ in range(batch_size)
    ], dtype=tf.float32)
