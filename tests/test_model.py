import pytest
import torch

from openunmix import model
from openunmix import umxse
from openunmix import umxhq
from openunmix import umx
from openunmix import umxl


@pytest.fixture(params=[10, 100])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 5])
def nb_samples(request):
    return request.param


@pytest.fixture(params=[111, 1024])
def nb_bins(request):
    return request.param


@pytest.fixture
def spectrogram(request, nb_samples, nb_channels, nb_bins, nb_frames):
    return torch.rand((nb_samples, nb_channels, nb_bins, nb_frames))


@pytest.fixture(params=[True, False])
def unidirectional(request):
    return request.param


@pytest.fixture(params=[32])
def hidden_size(request):
    return request.param


def test_shape(spectrogram, nb_bins, nb_channels, unidirectional, hidden_size):
    unmix = model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=nb_channels,
        unidirectional=unidirectional,
        nb_layers=1,  # speed up training
        hidden_size=hidden_size,
    )
    unmix.eval()
    Y = unmix(spectrogram)
    assert spectrogram.shape == Y.shape


@pytest.mark.parametrize("model_fn", [umx, umxhq, umxse, umxl])
def test_model_loading(model_fn):
    X = torch.rand((1, 2, 4096))
    model = model_fn(niter=0, pretrained=True)
    Y = model(X)
    assert Y[:, 0, ...].shape == X.shape
