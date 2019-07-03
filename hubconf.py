import utils
import torch.hub
from model import OpenUnmix


# Optional list of dependencies required by the package
dependencies = ['torch', 'numpy']


def OpenUnmixStereo(
    target='vocals', device='cpu', pretrained=True, *args, **kwargs
):
    """
    Stereo BiLSTM Model trained on MUSDB18-WAV 
    """
    # set urls for weights
    target_urls = {
        'bass': 'https://open-unmix.s3.eu-west-3.amazonaws.com/pytorch-models/OpenUnmix-1.0/bass-e3e1c4be.pth',
        'drums': 'https://open-unmix.s3.eu-west-3.amazonaws.com/pytorch-models/OpenUnmix-1.0/drums-a849ec7d.pth',
        'other': 'https://open-unmix.s3.eu-west-3.amazonaws.com/pytorch-models/OpenUnmix-1.0/other-732546c5.pth',
        'vocals': 'https://open-unmix.s3.eu-west-3.amazonaws.com/pytorch-models/OpenUnmix-1.0/vocals-f3d5fab0.pth'
    }

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(
        rate=44100,
        n_fft=4096,
        bandwidth=16000
    )

    # load open unmix model
    unmix = OpenUnmix(
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        hidden_size=512,
        max_bin=max_bin
    )

    # enable centering of stft to minimize reconstruction error
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            target_urls[target],
            map_location=device
        )['state_dict']
        unmix.load_state_dict(state_dict)
        unmix.stft.center = True
        unmix.eval()

    return unmix.to(device)
