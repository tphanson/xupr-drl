import os
import tensorflow as tf
from tensorflow import keras
from tf_agents.trajectories import policy_step, trajectory, time_step

from helper.utils import parse_experiences

# Saving dir
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')


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
        self.step = tf.Variable(initial_value=0, dtype=tf.int32, name='step')
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

    def _data_spec(self):
        return trajectory.from_transition(
            self.time_step_spec,
            policy_step.PolicyStep(action=self.action_spec, state=(), info=()),
            self.time_step_spec,
        )

    def _policy(self):
        # Define I/O
        image_shape = self.observation_spec.shape
        num_of_actions = self.action_spec.maximum - self.action_spec.minimum + 1
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
            keras.layers.Dense(768, activation='relu'),
        ])
        rnn = None
        head = keras.Sequential([
            keras.layers.Dense(num_of_actions),
            keras.layers.Softmax(),
        ])
        # Flow data
        x = cnn(inputs)
        x = head(x)
        # Return model
        return keras.Model(inputs=inputs, outputs=x)

    def _explore(self, actions):
        exploring = tf.cast(tf.greater(
            tf.random.uniform(actions.shape, minval=0, maxval=1),
            tf.fill(actions.shape, self.epsilon),
        ), dtype=tf.int32)
        random_actions = tf.random.uniform(
            actions.shape,
            minval=self.action_spec.minimum,
            maxval=self.action_spec.maximum,
            dtype=tf.int32
        )
        actions = exploring * random_actions + (1 - exploring) * actions
        return actions

    def action(self, ts):
        q_values = self.policy(ts.observation)
        actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
        actions = self._explore(actions)
        return policy_step.PolicyStep(action=actions, state=(), info=())

    @tf.function
    def _loss(self):
        return 0

    @tf.function
    def _train_step(self, step_types, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            q_values = tf.gather_nd(
                self.policy(states),
                actions,
                batch_dims=1
            )
            next_q_values = tf.reduce_max(
                self.policy(next_states),
                axis=1
            )
            not_last = tf.cast(
                tf.less(
                    step_types,
                    time_step.StepType.LAST
                ),
                dtype=tf.float32
            )
            q_targets = rewards + self.discount * next_q_values * not_last
            loss = tf.reduce_mean(tf.square(q_values - q_targets))
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
        if self.step % 1000 == 0:
            self._save_checkpoint()
        return loss

    def _load_checkpoint(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)

    def _save_checkpoint(self):
        self.manager.save()

    def get_step(self):
        return int(self.step.numpy())
