import torch
import numpy as np
import json
from pathlib import Path
import openunmix 
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



if __name__ == "__main__":
    umx = openunmix.umxhq_spec(device="cpu")

    for target in ['vocals', 'drums', 'bass', 'other']:
        x = torch.randn(
            (1, 2, 2049, 128),
            requires_grad=True
        )
        umx_t = umx[target]
        out = umx_t(x)

        # Export the model
        # using nb_frames, nb_samples, nb_channels, nb_bins
        torch.onnx.export(
            umx_t,                 # model being run
            x,                   # model input (or a tuple for multiple inputs)
            target + ".onnx",
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=9,     # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['mix'],       # the model's input names
            output_names=[target],   # the model's output names
            verbose=True
        )

        onnx_model = onnx.load(target + ".onnx")  # load onnx model
        print(onnx_model.graph.input[0])
        # onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = -1
        tf_rep = prepare(onnx_model)  # prepare tf representation
        tf_rep.export_graph(target + ".pb")  # export the model

    umx = openunmix.umxse_spec(device="cpu")

    for target in ['speech']:
        x = torch.randn(
            (1, 1, 513, 128),
            requires_grad=True
        )
        umx_t = umx[target]
        out = umx_t(x)

        # Export the model
        # using nb_frames, nb_samples, nb_channels, nb_bins
        torch.onnx.export(
            umx_t,                 # model being run
            x,                   # model input (or a tuple for multiple inputs)
            target + ".onnx",
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=9,     # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['mix'],       # the model's input names
            output_names=[target],   # the model's output names
            # dynamic_axes={'mix': {3: 'sequence'}, target: {3: 'sequence'}},
            verbose=True
        )

        onnx_model = onnx.load(target + ".onnx")  # load onnx model
        print(onnx_model.graph.input[0])
        # onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = -1
        tf_rep = prepare(onnx_model)  # prepare tf representation
        tf_rep.export_graph(target + ".pb")  # export the model
