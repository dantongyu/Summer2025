'''
Name: convert.py
Description: Convert between model formats, supports torch to mobile, onnx to tflite
Date: 2023-08-25
Last Modified: 2023-08-25
'''

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #this hides tensorflow output spam on import
import sys

from klogs import kLogger
TAG = "CONVERT"
log = kLogger(TAG)

import config
from config import CONFIG

if CONFIG["permute"]:
    log.info("Using permute")
    from models import get_model_permuteChannels as get_model
else:
    from models import get_model

def load_config(file):
    global CONFIG
    CONFIG = config.set_config(file)

def onnx_convert() -> None:
    '''
    Convert a pytorch model to an onnx model 

    Args:
        None
    Returns:
        None

    Examples:
        python convert.py --config config.json
    '''
    model_weights = torch.load(CONFIG["input_model"], map_location=torch.device("cpu"))['state']
    model = get_model(CONFIG["model"])().to('cpu')
    model.load_state_dict(model_weights)
    model.eval()
    inputs = torch.rand(1, 3, 224, 224)
    model.forward(inputs)
    onnx_program = torch.onnx.dynamo_export(model, inputs)
    onnx_program.save(CONFIG["output_model"])

def torch_to_android() -> None:
    '''
    Convert a pytorch model to an android model

    Args:
        None
    Returns:
        None

    Examples:
        python convert.py --config config.json
    '''

    model_weights = torch.load(CONFIG["input_model"], map_location=torch.device("cpu"))['state']
    model = get_model(CONFIG["model"])().to('cpu')
    model.load_state_dict(model_weights)
    model.filter = torch.nn.Identity()
    model.eval()
    inputs = torch.rand(1, 3, 224, 224)
    model.forward(inputs)
    traced_script_module = torch.jit.trace(model, inputs)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(CONFIG["output_model"])

def torch_to_mobile() -> None:
    '''
    Convert a pytorch model to a mobile model

    Args:
        None
    Returns:
        None

    Examples:
        python convert.py --config config.json
    '''
    model_weights = torch.load(CONFIG["input_model"], map_location=torch.device(CONFIG["device"]))['state']
    model = get_model(CONFIG["model"])().to('cpu')
    model.load_state_dict(model_weights)
    model.eval()
    log.info("Model loaded")
    if CONFIG["model"] == "gru":
        inputs = torch.rand(1, 3, 224, 224)
        hidden = torch.rand(2,500)
        model.forward(inputs,hidden)
        inputs_tensor = [
            coremltools.ImageType(
                name="image",
                shape=inputs.shape,
                scale=1/256.0,
                color_layout=coremltools.colorlayout.RGB
            ),
            coremltools.TensorType(
                name="hidden",
                shape=hidden.shape,
            ),
        ]
        outputs_tensor = [
            coremltools.TensorType(
                name="output",
            ),
            coremltools.TensorType(
                name="hidden_out",
            ),
        ]
        traced_model = torch.jit.trace(model, (inputs, hidden))

    else:
        inputs = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            model.forward(inputs)
        inputs_tensor = [
            coremltools.ImageType (
                name="image",
                shape=inputs.shape,
                scale=1/256.0,
                color_layout=coremltools.colorlayout.RGB
            )
        ]
        with torch.no_grad():
            traced_model = torch.jit.trace(model, torch.Tensor(inputs))
        outputs_tensor = None

    # Trace and export.
    pt_name = CONFIG["input_model"].split(".pth")[0] + ".pt"
    traced_model.save(pt_name)
    ml_model = coremltools.convert(
        model=pt_name,
        outputs=outputs_tensor,
        inputs=inputs_tensor,
        convert_to="mlprogram",
        debug=False,
    )
    ml_model.save(CONFIG["output_model"])

def onnx_to_tflite(onnx_model) -> None:
    '''
    Convert an onnx model to a tflite model

    Args:
        onnx_model (str): path to onnx model
        output_tflite (str): path to output tflite model

    Returns:
        None

    Examples:
        python convert.py --onnx_to_tflite -i model.onnx -o model.tflite
    '''
    tf_rep = prepare(onnx_model)

    with tempfile.TemporaryDirectory() as tmp:
        tf_rep.export_graph(f"{tmp}/model")
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{tmp}/model")
        # Convert the model
        tflite_model = converter.convert()

    # Save the model
    with open(CONFIG["output_model"], 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    '''
    Examples:
        To convert from onnx to tflite:
            python convert.py --onnx_to_tflite -i model.onnx -o model.tflite
        To convert from torch to mobile:
            python convert.py --torch_to_mobile -i model.pt -o model.ptl --model_name resnet34
    '''
    argparser = argparse.ArgumentParser(description='Convert between model formats')
    argparser.add_argument('-l','--level', type=str, default='info', help='Level of debug statement can be \
                           info, debug, warning, error, critical')
    argparser.add_argument('--verbose', action='store_true', help='Verbose output')
    argparser.add_argument('-c', '--config', type=str, default=None, help='Path to config file if not config.json')

    args = argparser.parse_args()
    if args.config:
        load_config(args.config)
    log.setLevel(args.level)

    if CONFIG["conversion"] == "onnx_to_tflite":
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx
        import tempfile
        try:
            onnx_to_tflite(onnx.load(CONFIG["input_model"]))
        except ImportError:
            log.error("Please install onnx and tensorflow packages")
            sys.exit(1)

    elif CONFIG["conversion"] == "onnx":
        import torch
        try: 
            onnx_convert()
        except ImportError as e:
            log.error(e)
            sys.exit(1)

    elif CONFIG["conversion"] == "coreml":
        import torch
        import coremltools
        try:
            torch_to_mobile()
        except ImportError as e:
            log.error(e)
            log.error("Please install torch and torchvision packages")
            sys.exit(1)

    elif CONFIG["conversion"] == "android":
        import torch
        from torch.utils.mobile_optimizer import optimize_for_mobile
        try:
            torch_to_android()
        except ImportError as e:
            log.error(e)
            log.error("Please install torch and torchvision packages")
            sys.exit(1)
    else:
        log.error("Unknown conversion type")
        log.error("Please set conversion type in config.json")
        log.error("Valid conversion types are: onnx_to_tflite, coreml")
        sys.exit(1)
