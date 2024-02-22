import os
import math
import numpy as np
import settings as settings
from importlib.util import find_spec
from src.vectors import Vector3

if find_spec('tflite_runtime'):
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

from src.vecna import Vecna
from src.training.functional import normalize_max
from src.log import logger


_model_path = os.path.join(settings.MODEL_FOLDER,"kinematics3.tflite")

_interpreter = Interpreter(_model_path)
_interpreter.allocate_tensors()

logger.info(f"Kinematics inference model loaded successfully")

    
def _infer(position: np.ndarray) -> np.ndarray:
    input_data = position.reshape(1,-1)
    input_data = normalize_max(input_data, Vecna.training.maxs).astype(np.float32)
    input_details = _interpreter.get_input_details()
    output_details = _interpreter.get_output_details()

    _interpreter.set_tensor(input_details[0]['index'], input_data)
    _interpreter.invoke()

    output_data = _interpreter.get_tensor(output_details[0]['index'])
    return output_data

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
    output = _infer(position)
    angles = np.floor((output*90)[0]).astype(int)
    return angles

    