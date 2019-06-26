import utils
import torch.hub

# Optional list of dependencies required by the package
dependencies = ['torch', 'numpy']


def OpenUnmix(target='vocals', *args, **kwargs):
    from model import OpenUnmix

    max_bin = utils.bandwidth_to_max_bin(
        44100,
        4096,
        1024
    )

    unmix = OpenUnmix(
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        hidden_size=512,
        max_bin=max_bin
    )

    target_urls = {
        'bass': 'https://www.dropbox.com/s/4q32250d11c8vzb/drums.pth.tar?dl=1',
        'drums': 'https://www.dropbox.com/s/qin2ctc669hd519/bass.pth.tar?dl=1',
        'other': 'https://www.dropbox.com/s/cptrjihxrmw3nv4/other.pth.tar?dl=1',
        'vocals': 'https://www.dropbox.com/s/qlmvf1ys0jbh5an/vocals.pth.tar?dl=1'
    }

    unmix.load_state_dict(
        torch.hub.load_state_dict_from_url(target_urls[target])
    )

    return unmix
