from random import random
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from env.objs import plane, ohmni, obstacle

VELOCITY_COEFFICIENT = 10


class Env:
    def __init__(self, gui=False, num_of_obstacles=20, dst_rad=3, image_shape=(96, 96)):
        # Env constants
        self.gui = gui
        self.timestep = 0.1
        self._left_wheel_id = 0
        self._right_wheel_id = 1
        # Env specs
        self.image_shape = image_shape
        self.num_of_obstacles = num_of_obstacles
        self.dst_rad = dst_rad
        self.destination = np.array([3, 0], dtype=np.float32)
        # Init
        self.client_id = self._init_ws()

    def _init_ws(self):
        """
        Create server and start, there are two modes:
        1. GUI: it visualizes the environment and allow controlling
            ohmni via sliders.
        2. Headless: by running everything in background, it's suitable
            for ai/ml/rl development.
        """
        # Init server
        client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep, physicsClientId=client_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        # Return
        return client_id

    def _randomize_destination(self):
        x = random() * self.dst_rad * (-1 if random() > 0.5 else 1)
        y = random() * self.dst_rad * (-1 if random() > 0.5 else 1)
        # vibe = random() * 4 * (-1 if random() > 0.5 else 1) # Level 1
        # destination = np.array([5, vibe], dtype=np.float32) # Level 1
        destination = np.array([x, y], dtype=np.float32)  # Level 2
        p.addUserDebugLine(
            np.append(destination, 0.),  # From
            np.append(destination, 3.),  # To
            [1, 0, 0],  # Red
            physicsClientId=self.client_id
        )
        return destination

    def _build(self):
        """ Including plane, ohmni, obstacles into the environment """
        # Add gravity
        p.setGravity(0, 0, -10, physicsClientId=self.client_id)
        # Add plane and ohmni
        plane(self.client_id)
        ohmni_id, _capture_image = ohmni(self.client_id)
        # Add obstacles at random positions
        # vibe = random() * 1.5 * (-1 if random() > 0.5 else 1) # Level 1
        # obstacle(self.client_id, pos=[3+vibe, 0, 0.5]) # Level 1
        for _ in range(self.num_of_obstacles):  # Level 2
            obstacle(self.client_id, avoids=[
                     [0, 0], self.destination])  # Level 2
        # Return
        return ohmni_id, _capture_image

    def _reset(self):
        """ Remove all objects, then rebuild them """
        p.resetSimulation(physicsClientId=self.client_id)
        self.destination = self._randomize_destination()
        self.ohmni_id, self._capture_image = self._build()

    def capture_image(self):
        """ Get image from navigation camera """
        if self._capture_image is None:
            raise ValueError('_capture_image is undefined')
        return self._capture_image(self.image_shape)

    def getContactPoints(self):
        """ Get Ohmni contacts """
        return p.getContactPoints(self.ohmni_id, physicsClientId=self.client_id)

    def getBasePositionAndOrientation(self):
        """ Get Ohmni position and orientation """
        return p.getBasePositionAndOrientation(self.ohmni_id, physicsClientId=self.client_id)

    def reset(self):
        """ Reset the environment """
        self._reset()

    def step(self, action):
        """ Controllers for left/right wheels which are separate """
        # Normalize velocities
        [left_wheel, right_wheel] = action
        left_wheel = left_wheel * VELOCITY_COEFFICIENT
        right_wheel = right_wheel * VELOCITY_COEFFICIENT
        # Step
        p.setJointMotorControl2(self.ohmni_id, self._left_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel,
                                physicsClientId=self.client_id)
        p.setJointMotorControl2(self.ohmni_id, self._right_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel,
                                physicsClientId=self.client_id)
        p.stepSimulation(physicsClientId=self.client_id)


class PyEnv(py_environment.PyEnvironment):
    def __init__(self, gui=False, image_shape=(96, 96)):
        super(PyEnv, self).__init__()
        # Parameters
        self.image_shape = image_shape
        self.input_shape = self.image_shape + (4,)
        self.max_steps = 500
        self._fix_vanish_hyperparam = 0.15
        self._num_of_obstacles = 25
        self._dst_rad = 6
        # Actions
        self._num_values = 5
        self._values = np.linspace(-1, 1, self._num_values)
        self._actions = np.transpose([
            np.tile(self._values, self._num_values),
            np.repeat(self._values, self._num_values)
        ])
        self._num_actions = len(self._actions)
        # PyEnvironment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0,
            maximum=self._num_actions - 1,
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.input_shape, dtype=np.float32,
            minimum=0,
            maximum=1,
            name='observation')
        # Init bullet server
        self._env = Env(
            gui,
            num_of_obstacles=self._num_of_obstacles,
            dst_rad=self._dst_rad,
            image_shape=self.image_shape
        )
        # Internal states
        self._state = None
        self._episode_ended = False
        self._num_steps = 0
        # Reset
        self._reset()

    def _get_image_state(self):
        _, _, rgb_img, _, seg_img = self._env.capture_image()
        img = np.array(rgb_img, dtype=np.float32) / 255
        # We add a constant to fix the problem of black pixels which vanish all the parameters
        mask = np.minimum(
            seg_img + self._fix_vanish_hyperparam,
            1 - self._fix_vanish_hyperparam,
            dtype=np.float32)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        return img, mask

    def _get_pose_state(self):
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        destination_posistion = np.append(self._env.destination, 0.)
        rotation = R.from_quat(
            [-orientation[0], -orientation[1], -orientation[2], orientation[3]])
        rel_position = rotation.apply(destination_posistion - position)
        _pose = rel_position[0:2]
        cosine_sim = np.dot([1, 0], _pose) / \
            (np.linalg.norm([1, 0]) * np.linalg.norm(_pose))
        return _pose.astype(dtype=np.float32), cosine_sim

    def _is_finished(self):
        """ Compute the distance from agent to destination """
        position, _ = self._env.getBasePositionAndOrientation()
        position = np.array(position[0:2], dtype=np.float32)
        distance = np.linalg.norm(position - self._env.destination)
        return distance < 0.5

    def _is_fatal(self):
        """ Predict a fall """
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        # Ohmni exceeds the number of steps
        if self._num_steps > self.max_steps:
            return True
        # Ohmni felt out of the environment
        if abs(position[2]) >= 0.5:
            return True
        # Ohmni is falling down
        if abs(orientation[0]) > 0.2 or abs(orientation[1]) > 0.2:
            return True
        return False

    def _is_collided(self):
        """ Predict collisions """
        collision = self._env.getContactPoints()
        for contact in collision:
            # Contact with things different from floor
            if contact[2] != 0:
                return True
        return False

    def _compute_reward(self):
        """ Compute reward and return (<stopped>, <reward>) """
        # Reaching the destination
        pose, cosine_sim = self._get_pose_state()
        if self._is_finished():
            return True, 10
        # Dead
        if self._is_fatal():
            return True, -10
        # Colliding
        if self._is_collided():
            return False, -0.1
        # Ohmni on his way
        return False, (cosine_sim - min(1, np.linalg.norm(pose)/10))/20

    def _reset(self):
        """ Reset environment"""
        self._env.reset()
        self._state = None
        self._episode_ended = False
        self._num_steps = 0
        self.set_state()
        return ts.restart(self._state)

    def action_spec(self):
        """ Return action specs """
        return self._action_spec

    def observation_spec(self):
        """ Return observation specs """
        return self._observation_spec

    def get_info(self):
        return {}

    def get_state(self):
        return self._state

    def set_state(self, state=None):
        # Gamifying
        (h, w) = self.image_shape
        _, mask = self._get_image_state()  # Image state
        pose, _ = self._get_pose_state()  # Pose state
        cent = np.array([w / 2, h / 2], dtype=np.float32)
        dest = -pose * 32 + cent  # Transpose/Scale/Tranform
        color = min(10, np.linalg.norm(pose))/20 + 0.25  # [0.25, 0.75]
        mask = cv.line(mask,
                       (int(cent[1]), int(cent[0])),
                       (int(dest[1]), int(dest[0])),
                       (color, color, color), thickness=3)
        observation = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        observation = np.reshape(observation, self.image_shape + (1,))
        # Set state
        if self._state is None:
            init_state = observation
            (_, _, stack_channel) = self.input_shape
            for _ in range(stack_channel - 1):
                init_state = np.append(init_state, observation, axis=2)
            self._state = np.array(init_state, dtype=np.float32)
        self._state = self._state[:, :, 1:]
        self._state = np.append(self._state, observation, axis=2)

    def _step(self, action):
        """ Step, action is velocities of left/right wheel """
        # Reset if ended
        if self._episode_ended:
            return self.reset()
        self._num_steps += 1
        # Step the environment
        self._env.step(self._actions[action])
        done, reward = self._compute_reward()
        # Compute and save states
        self.set_state()
        self._episode_ended = done
        # Transition
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def render(self, mode='rgb_array'):
        """ Show video stream from navigation camera """
        img = self.get_state()

        drawed_img = np.copy(img)
        drawed_img = cv.cvtColor(drawed_img, cv.COLOR_RGB2BGR)
        drawed_img = cv.resize(drawed_img, (512, 512))
        cv.imshow('OhmniInSpace-v0', drawed_img)
        cv.waitKey(10)

        return img


def env(gui=False):
    """ Convert pyenv to tfenv """
    pyenv = PyEnv(gui=gui)
    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    return tfenv
