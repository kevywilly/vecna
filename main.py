import time
from src.controller import Controller
from src.vecna import Vecna


controller = Controller()


def home():
    global controller
    controller.goto_angles(Vecna.home)
    while 1:
        pass


def demo():
    global controller
    while 1:
        controller.set_target_angles(Vecna.home)
        controller.wait_for_target()

        time.sleep(2)

        controller.set_target_angles(Vecna.squat)
        controller.wait_for_target()

        time.sleep(1)
    

        controller.set_target_angles(Vecna.stretch)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target_angles(Vecna.home)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target_angles(Vecna.twist)
        controller.wait_for_target()

        time.sleep(1)

        controller.set_target_angles(Vecna.twist * -1)
        controller.wait_for_target()

        time.sleep(1)


home()
#demo()
