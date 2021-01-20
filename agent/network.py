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
        self.gamma = 0.9
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
        self.target_policy = self._policy()
        self._update_target_policy()
        # Multi-steps Learning
        self._n_steps = 5

    #
    # Common functions
    #

    def get_step(self):
        return int(self.step.numpy())

    def get_n_steps(self):
        return self._n_steps

    #
    # Deep Q-Learning
    #

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

    #
    # Double Q-Learning
    #

    def _update_target_policy(self):
        self.target_policy.set_weights(self.policy.get_weights())

    #
    # Multi-steps Learning
    #

    def _expected_return(self, step_types, rewards):
        (batch_size, _) = step_types.shape
        not_last = tf.reverse(tf.transpose(tf.stack(
            [tf.reduce_prod(tf.split(
                tf.cast(
                    tf.less(step_types, time_step.StepType.LAST),
                    dtype=tf.float32
                ),
                num_or_size_splits=[self._n_steps - 1 - i, i + 1],
                axis=-1
            )[0], axis=-1) for i in range(self._n_steps)]
        )), axis=[-1])
        prev_states_not_last, end_state_not_last = tf.split(
            not_last,
            num_or_size_splits=[self._n_steps - 1, 1],
            axis=-1
        )
        prev_states_discount, last_state_discount = tf.split(
            tf.stack(
                [[self.gamma**i for i in range(self._n_steps)] for _ in range(batch_size)]),
            num_or_size_splits=[self._n_steps - 1, 1],
            axis=-1
        )
        supports_batch = tf.stack(
            [self._supports for _ in range(batch_size)])
        discounted_rewards = tf.reduce_sum(
            prev_states_not_last * prev_states_discount * rewards,
            axis=-1,
            keepdims=True
        )
        last_return = end_state_not_last * last_state_discount * supports_batch
        return discounted_rewards + last_return

    #
    # Distributional Learning
    #

    def _align(self, x, q):
        # Fundamental computation
        clipped_x = tf.minimum(tf.maximum(
            x, self._min_q_value), self._max_q_value)
        delta_z = (self._max_q_value - self._min_q_value) / \
            (self._num_of_atoms - 1)
        b = (clipped_x - self._min_q_value) / delta_z
        l = tf.math.floor(b)
        u = tf.math.ceil(b)
        # Create indices masks
        (batch_size, _) = x.shape
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
        # Compare to get boolean (active nodes)
        bool_l = tf.cast(tf.equal(mask_i, mask_l), dtype=tf.float32)
        bool_u = tf.cast(tf.equal(mask_i, mask_u), dtype=tf.float32)
        # Compute ml, mu at active nodes
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

    #
    # Predict
    #

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

    def action(self, ts, training=False):
        greedy_actions = self._greedy_action(ts.observation)
        (batch_size, _) = greedy_actions.shape
        print(greedy_actions.shape)
        greedy_actions = tf.reshape(greedy_actions, (batch_size,))
        actions = self._explore(greedy_actions)
        return policy_step.PolicyStep(action=actions, state=(), info=())

    #
    # Train
    #

    @tf.function
    def _loss(self, prediction, target):
        batch_loss = tf.reduce_sum(
            -tf.multiply(target, tf.math.log(prediction)),
            axis=-1
        )
        loss = tf.reduce_mean(batch_loss, axis=-1)
        return loss

    @tf.function
    def _train_step(self, step_types, start_state, action, rewards, end_state):
        with tf.GradientTape() as tape:
            z = self.policy(start_state)
            p = tf.gather_nd(z, action, batch_dims=1)
            optiomal_actions = self._greedy_action(end_state)
            next_z = self.target_policy(end_state)
            q = tf.gather_nd(next_z, optiomal_actions, batch_dims=1)
            x = self._expected_return(step_types, rewards)
            m = self._align(x, q)
            loss = self._loss(p, m)
        variables = self.policy.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, experiences):
        self.step.assign_add(1)
        step_types, start_state, action, rewards, end_state = parse_experiences(
            experiences, self._n_steps)
        loss = self._train_step(
            step_types, start_state, action, rewards, end_state)
        if self.step % self._callback_period == 0:
            self._save_checkpoint()
            self._update_target_policy()
        return loss

    #
    # Save/Load models
    #

    def _load_checkpoint(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)

    def _save_checkpoint(self):
        self.manager.save()
