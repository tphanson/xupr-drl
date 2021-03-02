import pybullet as p
import numpy as np
from random import random, choice


def plane(clientId):
    planeId = p.loadURDF('plane.urdf', physicsClientId=clientId)
    return planeId


def obstacle(clientId, pos=None, dynamic=False, avoids=[]):
    static_obstacles = ['env/model/table/table.urdf',
                        'cube_no_rotation.urdf',
                        'sphere2.urdf']
    dynamic_obstacles = ['cube_rotate.urdf']
    obstacles = dynamic_obstacles if dynamic else static_obstacles
    zone_rad = 8  # Dropping zone
    safe_rad = 0.5  # Minimum radian
    while pos is None:
        x = random()*zone_rad * (-1 if random() > 0.5 else 1)
        y = random()*zone_rad * (-1 if random() > 0.5 else 1)
        ok = True
        for (avoid_x, avoid_y) in avoids:
            if abs(x - avoid_x) < safe_rad or abs(y - avoid_y) < safe_rad:
                ok = False
                break
        pos = [x, y, 0.5] if ok else None
    return p.loadURDF(choice(obstacles), pos, physicsClientId=clientId)


def ohmni(clientId):
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    ohmniId = p.loadURDF('env/model/ohmni.urdf', start_pos,
                         start_orientation, physicsClientId=clientId)

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=80.0, aspect=1.0, nearVal=0.01, farVal=50)

    def get_nav_image(img_shape=(96, 96)):
        # Center of mass position and orientation (of tube)
        tube_position, tube_orientation, _, _, _, _ = p.getLinkState(
            ohmniId, 4,
            computeForwardKinematics=True,
            physicsClientId=clientId
        )
        tube_position = np.array(tube_position, dtype=np.float)
        rotation = np.array(p.getMatrixFromQuaternion(
            tube_orientation), dtype=np.float).reshape(3, 3)
        # Initial vectors
        init_eye_pos = np.array([0.015, 0, 0.4], dtype=np.float)
        init_target_pos = np.array([0.1, 0, -1], dtype=np.float)
        init_up_vector = np.array([0, 0, 1], dtype=np.float)
        # Rotated vectors
        eye_pos = tube_position + np.dot(rotation, init_eye_pos)
        target_pos = eye_pos + np.dot(rotation, init_target_pos)
        up_vector = np.dot(rotation, init_up_vector)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector,
            physicsClientId=clientId
        )
        width, height = img_shape
        return p.getCameraImage(
            width, height,
            view_matrix, projection_matrix,
            physicsClientId=clientId
        )

    return ohmniId, get_nav_image
