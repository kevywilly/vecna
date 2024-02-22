import atexit

import numpy as np
import traitlets
from adafruit_servokit import ServoKit
from traitlets import HasTraits

from src.kinematics import forward, inverse
from src.log import logger
from src.vecna import Vecna
from src.vectors import Pos3, Pose


def get_step(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


stepfunc = np.vectorize(get_step)


class Controller(HasTraits):
    step_size = traitlets.Int(default_value=5).tag(config=True)

    def __init__(self):

        try:
            self.kit = ServoKit(channels=16)
        except Exception as ex:
            logger.error(f'Could not load ServoKit: {ex.__str__()}')

        self.pose = Pose()
        self.pose.target = Vecna.home
        self.goto(Vecna.home)

        atexit.register(self.shutdown)

    def at_target(self):
        return self.pose.at_target()

    def apply_position(self, angles: np.array):

        servo_angles = (angles + Vecna.adj) * Vecna.dir + 90

        for i in range(4):
            for j in range(3):
                angle = servo_angles[i][j]
                pin = Vecna.joints[i][j]
                self.kit.servo[pin].angle = servo_angles

        self.pose.angles = angles
        self.pose.servo_angles = servo_angles
        self.pose.position = forward(angles)

    def goto(self, target: np.ndarray):
        self.apply_position(target)
        self.pose.target = target

    def goto_position(self, position: np.ndarray):
        self.goto(self._pose_to_angles(position))

    def _position_to_angles(self, pose: np.ndarray):
        angles = np.zeros((4, 3))

        for idx, pos3 in enumerate(pose):
            np_angles = inverse(Pos3(*pos3)).numpy()
            angles[idx] = np_angles

        return angles

    def set_target_position(self, position: np.ndarray):
        self.set_target(self._position_to_angles(position))

    def set_target(self, target: np.ndarray):
        self.pose.target = target

    def step(self):
        # logger.info(f'iterations: {self.iterations}')

        if self.at_target():
            return

        new_angles = self.pose.angles

        for i in range(int(self.step_size)):
            diff = self.pose.target - new_angles
            step = stepfunc(diff)
            new_angles = new_angles + step

        self.goto(new_angles)

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
    controller.set_target(Vecna.squat)
    while not controller.at_target():
        controller.step()
