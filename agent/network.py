import os
import tensorflow as tf
from tensorflow import keras
from tf_agents.trajectories import policy_step, trajectory, time_step

from helper.utils import parse_experiences, build_mask

# Saving dir
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')


class Network():
    def __init__(self, time_step_spec, observation_spec, action_spec):
        # Specs
        self.time_step_spec = time_step_spec
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.data_spec = self._data_spec()
        # Training params
        self.epsilon = 0.9
        self.discount = 0.9
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00001)
        self._callback_period = 1000
        self.step = tf.Variable(initial_value=0, dtype=tf.int32, name='step')
        # Deep Q-Learning
        self._num_of_actions = self.action_spec.maximum - self.action_spec.minimum + 1
        # Distributional Learning (C51)
        self._num_of_atoms = 51
        self._min_q_value = -3
        self._max_q_value = 1
        self._supports = tf.linspace(
            tf.constant(self._min_q_value, dtype=tf.float32),
            tf.constant(self._max_q_value, dtype=tf.float32),
            self._num_of_atoms
        )
        # Policies
        self.policy = self._policy()
        self.target_policy = self._policy()
        # Checkpoints
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.policy,
            step=self.step,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            CHECKPOINT_DIR,
            max_to_keep=1
        )
        self._load_checkpoint()
        # Double Q-Learning
        self._update_target_policy()

    """
    Common functions
    """

    def get_step(self):
        return int(self.step.numpy())

    """
    Deep Q-Learning
    """

    def _data_spec(self):
        return trajectory.from_transition(
            self.time_step_spec,
            policy_step.PolicyStep(action=self.action_spec, state=(), info=()),
            self.time_step_spec,
        )

    def _policy(self):
        # Define I/O
        image_shape = self.observation_spec.shape
        # Define network
        inputs = keras.layers.Input(shape=image_shape)
        cnn = keras.Sequential([  # (96, 96, *)
            keras.layers.Conv2D(  # (92, 92, *)
                filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (46, 46, *)
            keras.layers.Conv2D(  # (42, 42, 32)
                filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (21, 21, *)
            keras.layers.Conv2D(  # (10, 10, *)
                filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (5, 5, *)
            keras.layers.Flatten(),
            keras.layers.Dense(2048, activation='relu'),
        ])
        rnn = None
        head = keras.Sequential([
            keras.layers.Dense(self._num_of_actions * self._num_of_atoms),
            keras.layers.Reshape((self._num_of_actions, self._num_of_atoms)),
            keras.layers.Softmax(),
        ])
        # Flow data
        x = cnn(inputs)
        x = head(x)
        # Return model
        return keras.Model(inputs=inputs, outputs=x)

    """
    Double Q-Learning
    """

    def _update_target_policy(self):
        self.target_policy.set_weights(self.policy.get_weights())

    """ Distributional Learning """

    def _align(self, x, q, batch_size):
        # Fundamental computation
        clipped_x = tf.minimum(tf.maximum(
            x, self._min_q_value), self._max_q_value)
        delta_z = (self._max_q_value - self._min_q_value) / \
            (self._num_of_atoms - 1)
        b = (clipped_x - self._min_q_value) / delta_z
        l = tf.math.floor(b)
        u = tf.math.ceil(b)
        # Create indices masks
        mask_i = build_mask(batch_size, self._num_of_atoms)
        mask_l = tf.repeat(
            tf.expand_dims(l, axis=1),
            repeats=[self._num_of_atoms],
            axis=1
        )
        mask_u = tf.repeat(
            tf.expand_dims(u, axis=1),
            repeats=[self._num_of_atoms],
            axis=1
        )
        # Compare to get boolean (active node)
        bool_l = tf.cast(tf.equal(mask_i, mask_l), dtype=tf.float32)
        bool_u = tf.cast(tf.equal(mask_i, mask_u), dtype=tf.float32)
        # Compute ml at active nodes
        _ml = tf.repeat(
            tf.expand_dims(q * (u - b), axis=1),
            repeats=[self._num_of_atoms],
            axis=1
        )
        ml = tf.reduce_sum(tf.multiply(bool_l, _ml), axis=-1)
        _mu = tf.repeat(
            tf.expand_dims(q * (b - l), axis=1),
            repeats=[self._num_of_atoms],
            axis=1
        )
        mu = tf.reduce_sum(tf.multiply(bool_u, _mu), axis=-1)
        # Return aligned distribution
        return mu + ml

    """
    Predict
    """

    def _greedy_action(self, observation):
        distributions = self.target_policy(observation)
        transposed_x = tf.reshape(self._supports, (self._num_of_atoms, 1))
        q_values = tf.matmul(distributions, transposed_x)
        actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
        return actions

    def _explore(self, greedy_actions):
        exploring = tf.cast(tf.greater(
            tf.random.uniform(greedy_actions.shape, minval=0, maxval=1),
            tf.fill(greedy_actions.shape, self.epsilon),
        ), dtype=tf.int32)
        random_actions = tf.random.uniform(
            greedy_actions.shape,
            minval=self.action_spec.minimum,
            maxval=self.action_spec.maximum,
            dtype=tf.int32
        )
        actions = exploring * random_actions + (1 - exploring) * greedy_actions
        return actions

    def action(self, ts):
        greedy_actions = self._greedy_action(ts.observation)
        (batch_size, _) = greedy_actions.shape
        greedy_actions = tf.reshape(greedy_actions, (batch_size,))
        actions = self._explore(greedy_actions)
        return policy_step.PolicyStep(action=actions, state=(), info=())

    """
    Train
    """

    # @tf.function
    def _loss(self, prediction, target):
        batch_loss = tf.reduce_sum(
            -tf.multiply(target, tf.math.log(prediction)),
            axis=-1
        )
        loss = tf.reduce_mean(batch_loss, axis=-1)
        return loss

    # @tf.function
    def _train_step(self, step_types, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            (batch_size,) = step_types.shape
            z = self.policy(states)
            p = tf.gather_nd(z, actions, batch_dims=1)
            optiomal_actions = self._greedy_action(next_states)
            next_z = self.target_policy(next_states)
            q = tf.gather_nd(next_z, optiomal_actions, batch_dims=1)
            not_last = tf.reshape(
                tf.cast(
                    tf.less(step_types, time_step.StepType.LAST),
                    dtype=tf.float32
                ),
                (batch_size, 1)
            )
            supports_batch = tf.stack(
                [self._supports for _ in range(batch_size)])
            rewards = tf.reshape(rewards, (batch_size, 1))
            x = rewards + self.discount * supports_batch * not_last
            m = self._align(x, q, batch_size)
            loss = self._loss(p, m)
        variables = self.policy.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, experiences):
        # Increase training step
        self.step.assign_add(1)
        # Train
        step_types, states, actions, rewards, next_states = parse_experiences(
            experiences)
        loss = self._train_step(
            step_types, states, actions, rewards, next_states)
        if self.step % self._callback_period == 0:
            self._save_checkpoint()
            self._update_target_policy()
        return loss

    """
    Save/Load models
    """

    def _load_checkpoint(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)

    def _save_checkpoint(self):
        self.manager.save()
