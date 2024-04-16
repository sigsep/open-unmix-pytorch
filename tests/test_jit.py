import torch
import torch.onnx
import pytest

from openunmix import model


@pytest.mark.skip(reason="Currently not supported")
def test_onnx():
    """Test ONNX export of the separator

    currently results in erros, blocked by
    https://github.com/pytorch/pytorch/issues/49958
    """
    nb_samples = 1
    nb_channels = 2
    nb_timesteps = 11111

    example = torch.rand((nb_samples, nb_channels, nb_timesteps), device="cpu")
    # set model to eval due to non-deterministic behaviour of dropout
    umx = model.OpenUnmix(nb_bins=2049, nb_channels=2).eval().to("cpu")

    # creatr separator
    separator = (
        model.Separator(target_models={"source_1": umx, "source_2": umx}, niter=1, filterbank="asteroid")
        .eval()
        .to("cpu")
    )

    torch_out = separator(example)

    # Export the model
    torch.onnx.export(
        separator,
        example,
        "umx.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
