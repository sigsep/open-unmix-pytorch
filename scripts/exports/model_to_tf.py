import torch
import numpy as np
import json
from pathlib import Path
from openunmix import model
from openunmix import utils

import warnings
from contextlib import redirect_stderr
import io
import onnx
import warnings
from onnx_tf.backend import prepare
import pprint
pp = pprint.PrettyPrinter(indent=4)
import tensorflow as tf


def apply_remove_weight_norm(module):
    try:
        torch.nn.utils.remove_weight_norm(module)
    except ValueError:  # this module didn't have weight norm
        return


if __name__ == "__main__":

    umx = utils.load_target_models(
        targets="vocals",
        model_str_or_path="umxhq",
        device="cpu",
    )['vocals']

    x = torch.randn(
        (1, 2, 2049, 100),
        requires_grad=True
    )
    torch_out = umx(x)
    # Export the model
    # using nb_frames, nb_samples, nb_channels, nb_bins
    torch.onnx.export(
        umx,                 # model being run
        x,                   # model input (or a tuple for multiple inputs)
        "vocals.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=9,     # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['mix'],       # the model's input names
        output_names=['vocals'],   # the model's output names
        verbose=True,
        # dynamic_axes={'mix': {3: 'sequence'}, 'vocals': {3: 'sequence'}}
    )

    onnx_model = onnx.load("vocals.onnx")  # load onnx model
    print(onnx_model.graph.input[0])
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph("vocals-savedmodel")  # export the model
