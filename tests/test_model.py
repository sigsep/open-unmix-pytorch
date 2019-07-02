import pytest
import model
import torch


@pytest.fixture(params=[4096, 44100, 54321])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 5])
def nb_samples(request):
    return request.param


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


@pytest.fixture(params=[True, False])
def unidirectional(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_layers(request):
    return request.param


@pytest.fixture(params=[32, 512])
def hidden_size(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def n_fft(request):
    return request.param


@pytest.fixture(params=[2, 4])
def n_hop(request, n_fft):
    return n_fft // request.param


def test_shape(
    audio,
    nb_channels,
    unidirectional,
    nb_layers,
    hidden_size,
    n_fft,
    n_hop
):
    unmix = model.OpenUnmix(
        n_fft=n_fft,
        n_hop=n_hop,
        nb_channels=nb_channels,
        input_is_spectrogram=True,
        unidirectional=unidirectional,
        nb_layers=nb_layers,
        hidden_size=hidden_size
    )
    unmix.eval()
    spec = torch.nn.Sequential(unmix.stft, unmix.spec)
    X = spec(audio)
    Y = unmix(X)
    assert X.shape == Y.shape
