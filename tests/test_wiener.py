import pytest
import model
import torch
from filtering import wiener


@pytest.fixture(params=[10, 100])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[10, 127])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_sources(request):
    return request.param


@pytest.fixture
def target(request, nb_frames, nb_channels, nb_bins, nb_sources):
    return torch.rand((nb_frames, nb_bins, nb_channels, nb_sources))


@pytest.fixture
def mix(request, nb_frames, nb_channels, nb_bins):
    return torch.rand((nb_frames, nb_bins, nb_channels, 2))


def test_wiener_shape(target, mix):
    output = wiener(target, mix)
