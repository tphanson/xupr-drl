import tensorflow as tf


def parse_experiences(experiences, n_steps):
    step_types = experiences.step_type
    start_state, _, end_state = tf.split(
        experiences.observation,
        num_or_size_splits=[1, n_steps - 2, 1],
        axis=1
    )
    action, _ = tf.split(
        experiences.action,
        num_or_size_splits=[1, n_steps - 1],
        axis=1
    )
    rewards, _ = tf.split(
        experiences.reward,
        num_or_size_splits=[n_steps - 1, 1],
        axis=1
    )
    return step_types, tf.squeeze(start_state), action, rewards, tf.squeeze(end_state)


def build_mask(batch_size, num_of_atoms):
    return tf.constant([
        [
            [
                i for _ in range(num_of_atoms)
            ] for i in range(num_of_atoms)
        ] for _ in range(batch_size)
    ], dtype=tf.float32)
