import torch
from torch.testing._internal.jit_utils import JitTestCase
import model


class TestModels(JitTestCase):
    @staticmethod
    def _test_umx(self, device, check_export_import=True):
        example = torch.rand(
            (1, 2, 44100),
            device=device
        )
        # test model in eval due to non-deterministic behaviour of dropout
        umx = model.OpenUnmix().eval().to(device)
        self.checkTrace(umx, (example,), export_import=check_export_import)

        separator = model.Separator(targets={'test': umx}).eval().to(device)

        # disabble tracing for now as there are too many dynamic parts
        # self.checkTrace(separator, (example,), export_import=check_export_import)

        # instead test scripting
        torch.jit.script(separator)

    def test_umx(self):
        self._test_umx(self, device='cpu')
