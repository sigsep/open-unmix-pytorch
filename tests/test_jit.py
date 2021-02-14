import torch
import torch.onnx
import pytest
from torch.testing._internal.jit_utils import JitTestCase

from openunmix import model


class TestModels(JitTestCase):
    @staticmethod
    def _test_umx(self, device, check_export_import=True):
        nb_samples = 1
        nb_channels = 2
        nb_bins = 2049
        nb_frames = 11
        nb_timesteps = 4096 * 10

        example = torch.rand((nb_samples, nb_channels, nb_bins, nb_frames), device=device)
        # set model to eval due to non-deterministic behaviour of dropout
        umx = model.OpenUnmix(nb_bins=nb_bins, nb_channels=nb_channels).eval().to(device)

        # test trace
        self.checkTrace(umx, (example,), export_import=check_export_import)

        # creatr separator
        separator = (
            model.Separator(
                target_models={"source_1": umx, "source_2": umx}, niter=1, filterbank="asteroid"
            )
            .eval()
            .to(device)
        )

        example_time = torch.rand((nb_samples, nb_channels, nb_timesteps), device="cpu")

        # disable tracing check for now as there are too many dynamic parts
        self.checkTrace(separator, (example_time,), export_import=False, inputs_require_grads=False)
        # test scripting of the separator
        torch.jit.script(separator)

    def test_umx(self):
        self._test_umx(self, device="cpu")


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
        model.Separator(
            target_models={"source_1": umx, "source_2": umx}, niter=1, filterbank="asteroid"
        )
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
