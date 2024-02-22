import torch
import onnx_tf

import settings
import src.training.models as models
from torchvision.models import resnet50, resnet18
import tensorflow as tf
import onnx
from onnx import helper


torch_model_file = settings.model_path(f'kinematics3.pth')


def _file_out(file_in, ext):
    return f'{file_in.split(".")[0]}.{ext}'


def fix_onnx(file_in):
    onnx_model = onnx.load(file_in)
    file_out = _file_out(file_in, 'onnx')

    # Define a mapping from old names to new names
    name_map = {"input.1": "input_1", "onnx::Gemm_0":"onnx__Gemm_0"}

    # Initialize a list to hold the new inputs
    new_inputs = []

    # Iterate over the inputs and change their names if needed
    for inp in onnx_model.graph.input:
        if inp.name in name_map:
            # Create a new ValueInfoProto with the new name
            new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                    inp.type.tensor_type.elem_type,
                                                    [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)

    # Clear the old inputs and add the new ones
    onnx_model.graph.ClearField("input")
    onnx_model.graph.input.extend(new_inputs)

    # Go through all nodes in the model and replace the old input name with the new one
    for node in onnx_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]

    # Save the renamed ONNX model
    onnx.save(onnx_model, file_out)

    return file_out


def resnet18_2_onnx() -> str:
    model = resnet18(pretrained=True)
    file_out = resnet18.onnx
    #models.load_model(model, file_in)
    model.eval()

    # Set  input shape of the model
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)

    # Export  PyTorch model to ONNX format
    torch.onnx.export(model, dummy_input, file_out)

    return file_out


def torch_2_onnx(file_in) -> str:
    file_out = _file_out(file_in, 'onnx.tmp')
    model = models.Kinematics3(pretrained=True)
    model.eval()

    # Set  input shape of the model
    input_shape = (1, 3)
    dummy_input = torch.randn(input_shape)

    # Export  PyTorch model to ONNX format
    torch.onnx.export(model, dummy_input, file_out)

    return fix_onnx(file_out)


def onnx_2_tf(file_in: str) -> str:
    file_out = _file_out(file_in, 'tf')
    # Load  ONNX model
    onnx_model = onnx.load(file_in)

    # Convert ONNX model to TensorFlow format
    tf_model = onnx_tf.backend.prepare(onnx_model)

    # Export  TensorFlow  model
    tf_model.export_graph(file_out)

    return file_out


def tf_2_tflite(file_in: str):
    file_out = _file_out(file_in, 'tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(file_in)
    tflite_model = converter.convert()
    open(file_out, 'wb').write(tflite_model)
    return file_out


"""
onnx_file = convert_resnet('resnet.onnx')
fixed_onnx_file = fix_onnx(onnx_file, 'resnet_fixed.onnx')
tf_file = convert_tf(fixed_onnx_file, 'resnet.tf')
tflite_file = convert_tflite(tf_file, 'resnet.tflite')
"""


onnx_file = torch_2_onnx(torch_model_file)
tf_file = onnx_2_tf(onnx_file)
tflite_file = tf_2_tflite(tf_file)