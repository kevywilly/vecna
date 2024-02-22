import torch
import torch.nn as nn
import os
import settings as settings

_KINEMATICS3_FILENAME = 'kinematics3.pth'


class Kinematics3(nn.Sequential):
    def __init__(self, pretrained = False, filename=_KINEMATICS3_FILENAME):
        super().__init__(
            nn.Linear(3, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.pretrained = False
        self.filename = filename

        if pretrained:
            self.load_state()

    def load_state(self, filename=_KINEMATICS3_FILENAME):
        self.filename = filename
        path = os.path.join(settings.MODEL_FOLDER, filename)
        if not os.path.isfile(path):
            print('could not load model')
            return
        self.load_state_dict(torch.load(path))

    def save_state(self, filename=_KINEMATICS3_FILENAME):
        self.filename = filename
        torch.save(self.state_dict(),os.path.join(settings.MODEL_FOLDER, filename))

