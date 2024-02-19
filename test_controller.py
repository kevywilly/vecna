from src.controller import Controller


if __name__ == "__main__":
    controller = Controller()
    controller.set_target(controller.robot.squat)
    while not controller.at_target:
        controller.step()
