import pytest
import numpy as np
import model
import test


@pytest.fixture(params=[4094, 44100, 1249012])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 16])
def nb_samples(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def nfft(request):
    return int(request.param)


@pytest.fixture(params=[2, 4, 8])
def hop(request, nfft):
    return(nfft // request.param)


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return np.random.random((nb_samples, nb_channels, nb_timesteps))


def test_stft(audio, nb_channels, nfft, hop):
    unmix = model.OpenUnmix(nb_channels=nb_channels)
    X = unmix.spec(audio)
    X = X.detach().numpy()
    X_complex_np = X[..., 0] + X[..., 1]*1j
    out = test.istft(X_complex_np)

