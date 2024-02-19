from src.robot import robot
import numpy as np

def get_step(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

stepfunc = np.vectorize(get_step)
num_joints = 12

class Controller:

    def __init__(self):
        self.robot = robot
        self.position = robot.home
        self.target = robot.home
        self.step_size = 5
        self.goto(self.target)
        self.at_target=True
        self.iterations = 0

    def apply_position(self, target):
        target = (target * robot.dir) + robot.adj
        print(f'applied {target}')

    def goto(self, target: np.ndarray):
        self.apply_position(target)
        self.position = target
        #print(f'p/t: {self.position, self.target}')
       
    def set_target(self, target: np.ndarray):
        self.target = target
        self.at_target = False
        print(f'target: {self.target}')

    def step(self):
        print(f'iterations: {self.iterations}')
        self.at_target = (sum(sum(self.position == self.target)) == num_joints)
        if self.at_target:
            self.iterations = 0
            return
        
        new_pos = self.position

        for i in range(self.step_size):
            diff = self.target-new_pos
            step = stepfunc(diff)
            new_pos = new_pos + step
        
        self.iterations += 1
        self.goto(new_pos)
            
if __name__ == "__main__":
    controller = Controller()
    controller.set_target(robot.squat)
    while not controller.at_target:
        controller.step()