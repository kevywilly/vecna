import atexit
from importlib.util import find_spec

import numpy as np

if find_spec('adafruit_servokit'):
    from adafruit_servokit import ServoKit
else:
    from src.mocks.mock_servo_kit import MockServoKit as ServoKit

import traitlets

from src.kinematics_lite import forward, inverse
from src.log import logger
from src.vecna import Vecna
from src.vectors import Pose


def get_step(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


_step_func = np.vectorize(get_step)


def _positions_to_angles(positions: np.ndarray):
    angles = np.zeros((4, 3))

    for idx, position in enumerate(positions):
        np_angles = inverse(position)
        angles[idx] = np_angles

    return angles


class Controller(traitlets.HasTraits):
    step_size = traitlets.Int(default_value=5).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, kwargs)
        try:
            self.kit = ServoKit(channels=16)
        except Exception as ex:
            logger.error(f'Could not load ServoKit: {ex.__str__()}')

        self.pose = Pose()
        self.pose.target_angles = Vecna.home
        self.goto_angles(Vecna.home)

        atexit.register(self.shutdown)

    def at_target(self):
        return self.pose.at_target()

    def apply_angles(self, angles: np.array):

        servo_angles = (angles + Vecna.adj) * Vecna.dir + 90

        for i in range(4):
            for j in range(3):
                angle = servo_angles[i][j]
                pin = Vecna.joints[i][j]
                if self.kit is not None:
                    self.kit.servo[pin].angle = angle

        self.pose.angles = angles
        self.pose.servo_angles = servo_angles
        self.pose.position = forward(angles)

    def goto_angles(self, angles: np.ndarray):
        self.apply_angles(angles)
        self.pose.target_angles = angles

    def goto_positions(self, positions: np.ndarray):
        self.goto_angles(_positions_to_angles(positions))

    def set_target_positions(self, positions: np.ndarray):
        self.set_target_angles(_positions_to_angles(positions))

    def set_target_angles(self, target: np.ndarray):
        self.pose.target_angles = target

    def step(self):

        if self.at_target():
            return

        new_angles = self.pose.angles

        for i in range(int(self.step_size)):
            diff = self.pose.target_angles - new_angles
            step = _step_func(diff)
            new_angles = new_angles + step

        self.goto_angles(new_angles)

    def wait_for_target(self, delay=0.1):
        while not self.at_target():
            self.step()

    def stop(self):
        for i in range(4):
            for j in range(3):
                pin = Vecna.joints[i][j]
                self.kit.servo[pin].angle = None

    def shutdown(self):
        if self.kit:
            self.stop()


if __name__ == "__main__":
    controller = Controller()
    controller.set_target_angles(Vecna.squat)
    while not controller.at_target():
        controller.step()
