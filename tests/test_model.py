import pytest
import numpy as np
import model
import torch


@pytest.fixture(params=[4096, 44100, 1249012])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 16])
def nb_samples(request):
    return request.param


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_shape(audio, nb_channels):
    unmix = model.OpenUnmix(nb_channels=nb_channels, input_is_spectrogram=True)
    unmix.eval()
    spec = torch.nn.Sequential(unmix.stft, unmix.spec)
    X = spec(audio)
    Y = unmix(X)
    assert X.shape == Y.shape
