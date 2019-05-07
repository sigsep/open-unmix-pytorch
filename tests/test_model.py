import pytest
import numpy as np
import model


@pytest.fixture(params=[4094, 44100, 1249012])
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
    return np.random.random((nb_samples, nb_channels, nb_timesteps))


def test_model(audio, nb_channels):
    unmix = model.OpenUnmix(nb_channels=nb_channels)
    assert unmix(audio)


def test_shape(audio, nb_channels):
    unmix = model.OpenUnmix(nb_channels=nb_channels)
    X = unmix.spec(audio)
    Y = unmix(X)
    assert X.shape == Y.shape
