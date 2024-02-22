from abc import ABC

import numpy as np


class Vector3(ABC):
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def array(self):
        return [self.x, self.y, self.z]

    def numpy(self):
        return np.array(self.array())

    @classmethod
    def from_numpy(cls, array):
        return Vector3(*array)

    def __repr__(self):
        return f'({self.x},{self.y},{self.z})'


class Angle3(Vector3):
    pass


class Pos3(Vector3):
    pass


class Pose:
    num_links = 12

    def __init__(self):
        self.target_angles = np.zeros(4, 3)
        self.angles = np.zeros(4,3)
        self.position = np.zeros(4, 3)
        self.servo_angles = np.zeros(4,3)

    def at_target(self):
        return bool(sum(sum(self.angles == self.target_angles)) == self.num_links)
