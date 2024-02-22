import time
from src.controller import Controller
from src.vecna import Vecna
from src.kinematics import Kinematics, Vector3
import math

controller = Controller()

def home():
    global controller
    controller.goto(Vecna.home)
    target = Vecna.home
    time.sleep(1)
    target[0] =  Kinematics.inverse(Vector3(0,63,40)).array()
    controller.goto(target)    
    time.sleep(0.5)
    target[0] =  Kinematics.inverse(Vector3(-30,53,40)).array()
    controller.goto(target)
    time.sleep(0.5)
    target[0] =  Kinematics.inverse(Vector3(-30,53,0)).array()
    controller.goto(target)
    time.sleep(0.5)
    while 1:
        pass


def demo():
    global controller
    while 1:
        controller.set_target(Vecna.home)
        controller.wait_for_target()

        time.sleep(2)

        controller.set_target(Vecna.squat)
        controller.wait_for_target()

        time.sleep(1)
    

        controller.set_target(Vecna.stretch)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target(Vecna.home)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target(Vecna.twist)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target(Vecna.twist * -1)
        controller.wait_for_target()

        time.sleep(1)


home()
#demo()
