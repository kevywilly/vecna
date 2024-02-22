import os
import numpy as np
import settings as settings
from tflite_runtime.interpreter import Interpreter
from src.vecna import Vecna
from src.training.functional import normalize_max

model_path = os.path.join(settings.MODEL_FOLDER,"kinematics3.tflite")

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print('input_shape', input_shape)


position = np.array([0,65,0]).reshape(1,-1)
position = normalize_max(position, Vecna.training.maxs).astype(np.float32)
print('position', position)
input_data = position

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
output_data = np.floor(output_data*90).astype(int)[0]
print(output_data)