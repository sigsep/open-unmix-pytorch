import pytest
import numpy as np
import torch
import model
import test
import torchaudio


def istft(
    X,
    rate=44100,
    n_fft=4096,
    n_hopsize=1024,
    center=True
):
    return torchaudio.functional.istft(
        X,
        n_fft=n_fft,
        hop_length=n_hopsize,
        window=torch.hann_window(n_fft),
        center=center,
        normalized=False,
        onesided=True,
        pad_mode='reflect'
    )


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


@pytest.mark.parametrize(
    ("center"),
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="https://github.com/pytorch/audio/issues/500"
            )
        ),
    ],
)
def test_stft(audio, nb_channels, nfft, hop, center):
    unmix = model.OpenUnmix(nb_channels=nb_channels)
    unmix.stft.center = center
    X = unmix.stft(audio)
    X = X.detach()
    out = istft(X, center=center)
    assert np.sqrt(
        np.mean((audio.detach().numpy() - out.detach().numpy())**2)
    ) < 1e-6
