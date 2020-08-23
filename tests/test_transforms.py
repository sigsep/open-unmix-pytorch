import pytest
import numpy as np
import torch
import model


@pytest.fixture(params=[4096, 4096*10])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2])
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
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_stft(audio, nb_channels, nfft, hop):
    # we should only test for center=True as
    # False doesn't pass COLA
    # https://github.com/pytorch/audio/issues/500
    stft = model.STFT(n_fft=nfft, n_hop=hop, center=True)
    X = stft(audio)
    X = X.detach()
    out = model.istft(
        X,
        n_fft=nfft,
        n_hop=hop,
        center=True,
        window=stft.window,
        length=audio.shape[-1]
    )
    assert np.sqrt(
        np.mean((audio.detach().numpy() - out.detach().numpy())**2)
    ) < 1e-6
