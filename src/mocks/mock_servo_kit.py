class MockServoKit:

    def __init__(self, channels):
        self.channels = channels
        self.servo = []
        for index, i in enumerate(channels):
            self.servos.append(MockServo(index))

class MockServo:
    def __init__(self, pin):
        self.pin = pin
        self.angle = 0
