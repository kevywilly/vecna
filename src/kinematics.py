import math

from src.training.functional import normalize_max
from src.training.models import Kinematics3
from src.vecna import Vecna
import torch
import numpy as np

from src.vectors import Vector3


model = Kinematics3(pretrained=True)
model.eval()

def forward(angles: np.ndarray) -> np.ndarray:
    thetas = Vecna.dims.thetas
    diff = (angles - thetas)
    theta0,theta1,theta2 = diff

    hyp0 = math.cos(math.radians(theta0)) * Vecna.dims.coxa
    x0 = math.sin(math.radians(theta0)) * Vecna.dims.coxa
    z0 = 0

    x1 = 0
    hyp1 = math.cos(math.radians(theta1)) * Vecna.dims.femur
    z1 = math.sin(math.radians(theta1)) * Vecna.dims.femur

    hyp2 = math.cos(math.radians(theta2)) * Vecna.dims.tibia

    y = hyp0+hyp1+hyp2
    x = math.sin(math.radians(theta0)) * y
    z = z0+z1

    return Vector3(int(math.floor(x)), int(math.floor(y)), int(math.floor(z))).numpy()


def inverse(position: np.ndarray) -> np.ndarray:
    x = position.copy()
    x = normalize_max(x, Vecna.training.maxs)
    input = torch.from_numpy(x.astype(np.float32)).reshape(1,-1)
    output = model.forward(input)
    angles = np.floor((output.detach().numpy()*90)[0]).astype(int)
    return angles


