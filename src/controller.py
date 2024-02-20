from src.vecna import Vecna
from adafruit_servokit import ServoKit
import numpy as np
import traitlets
from traitlets import HasTraits
from src.log import logger
import atexit
import time

def get_step(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

stepfunc = np.vectorize(get_step)
num_joints = 12

class Controller(HasTraits):

    step_size = traitlets.Int(default_value = 5).tag(config=True)
    at_target = traitlets.Bool()
    moving = traitlets.Bool()


    def __init__(self):

        self.position = Vecna.home
        self.target = Vecna.home
        self.iterations = 0

        try:
            self.kit = ServoKit(channels=16)
        except Exception as ex:
            logger.error(f'Could not load ServoKit: {ex.__str__()}')

        # self.goto(self.target)
        self.stop()
        self.at_target = False
        self.moving = False 
        
        atexit.register(self.shutdown)


    def apply_position(self, target):
        
        angles = (target + Vecna.adj) * Vecna.dir + 90

        for i in range(4):
            for j in range(3):
                angle = angles[i][j]
                pin = Vecna.joints[i][j]
                self.kit.servo[pin].angle = angle

        #logger.info(f'applied {target}')

    def goto(self, target: np.ndarray):
        self.apply_position(target)
        self.position = target
       
    def set_target(self, target: np.ndarray):
        self.target = target
        self.at_target = False

    def step(self):
        #logger.info(f'iterations: {self.iterations}')
        self.at_target: bool = bool(sum(sum(self.position == self.target)) == num_joints)
        if self.at_target:
            self.iterations = 0
            self.moving = False
            return
        
        self.moving = True
        new_pos = self.position

        for i in range(self.step_size):
            diff = self.target-new_pos
            step = stepfunc(diff)
            new_pos = new_pos + step
        
        self.iterations += 1
        self.goto(new_pos)

    def wait_for_target(self, delay = 0.1):
        while not self.at_target:
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
    while not controller.at_target:
        
        controller.step()
