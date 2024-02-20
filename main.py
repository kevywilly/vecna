from src.controller import Controller
import time
if __name__ == "__main__":
    controller = Controller()
    # controller.set_target(controller.robot.squat)
    # while not controller.at_target:
    #    controller.step()
    
    # while 1:
    controller.set_target(controller.robot.squat)

    while not controller.at_target:
        controller.step()
        time.sleep(.01)

    controller.set_target(controller.robot.twist)

    time.sleep(1)
    while not controller.at_target:
        controller.step()
        time.sleep(.01)

    controller.stop()
