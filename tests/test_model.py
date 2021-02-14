import pytest
import torch

from openunmix import model


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
