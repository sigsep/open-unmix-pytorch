import torch
from torch.testing._internal.jit_utils import JitTestCase

from openunmix import model


class TestModels(JitTestCase):
    @staticmethod
    def _test_umx(self, device, check_export_import=True):
        nb_samples = 1
        nb_channels = 2
        nb_bins = 2049
        nb_frames = 11

        example = torch.rand(
            (nb_samples, nb_channels, nb_bins, nb_frames),
            device=device
        )
        # set model to eval due to non-deterministic behaviour of dropout
        umx = model.OpenUnmix(
            nb_bins=nb_bins,
            nb_channels=nb_channels
        ).eval().to(device)

        # test trace
        self.checkTrace(umx, (example,), export_import=check_export_import)

        # creatr separator
        separator = model.Separator(
            target_models={'source_1': umx, 'source_2': umx},
            niter=1
        ).eval().to(device)

        # disable tracing check for now as there are too many dynamic parts
        # self.checkTrace(separator, (example,), export_import=False)

        # test scripting of the separator
        torch.jit.script(separator)

    def test_umx(self):
        self._test_umx(self, device='cpu')
