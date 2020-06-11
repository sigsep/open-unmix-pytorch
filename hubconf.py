import utils
import torch.hub


# Optional list of dependencies required by the package
dependencies = ['torch', 'numpy']


def umxhq(
    target='vocals', device='cpu', pretrained=True, *args, **kwargs
):
    """
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18-HQ

    Args:
        target (str): select the target for the source to be separated.
                      Supported targets are
                        ['vocals', 'drums', 'bass', 'other']
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        device (str): selects device to be used for inference
    """
    # set urls for weights
    target_urls = {
        'bass': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/bass-8d85a5bd.pth',
        'drums': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/drums-9619578f.pth',
        'other': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/other-b52fbbf7.pth',
        'vocals': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/vocals-b62c91ce.pth'
    }

    from model import OpenUnmix

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
        )
        unmix.load_state_dict(state_dict)
        unmix.stft.center = True
        unmix.eval()

    return unmix.to(device)


def umx(
    target='vocals', device='cpu', pretrained=True, *args, **kwargs
):
    """
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18

    Args:
        target (str): select the target for the source to be separated.
                      Supported targets are
                        ['vocals', 'drums', 'bass', 'other']
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        device (str): selects device to be used for inference
    """
    # set urls for weights
    target_urls = {
        'bass': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/bass-646024d3.pth',
        'drums': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/drums-5a48008b.pth',
        'other': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/other-f8e132cc.pth',
        'vocals': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/vocals-c8df74a5.pth'
    }

    from model import OpenUnmix

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
        )
        unmix.load_state_dict(state_dict)
        unmix.stft.center = True
        unmix.eval()

    return unmix.to(device)


def separator(
    targets=['vocals', 'drums', 'bass', 'other'],
    model_name='umxhq', pretrained=True,
    residual=False, niter=1,
    device='cpu', *args, **kwargs
):
    """
    Create a complete Separator exploiting 2-channel/stereo BiLSTM UMX with
    several targets, trained on MUSDB18-HQ or MUSDB18.

    Args:
        targets (str): select the targets for the source to be separated.
                       a list including: ['vocals', 'drums', 'bass', 'other'].
                       If you don't pick them all, you probably want to
                       activate the `residual=True` option.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of postprocessing iterations to reduce interferences
        device (str): selects device to be used for inference
    """
    from model import Separator

    assert model_name in ['umx', 'umxhq'], 'model_name must be either '
                                           '`umx` or `umxhq`'
    load_fn = umx if model_name=='umx' else umxhq 

    # load targets models
    targets_models = {
        target:load_fn(target, device, pretrained, *args, **kwargs)
        for target in targets
    }

    # create the separator
    separator = Separator(targets=targets_models,
                          niter=1,
                          residual=residual,
                          out=None,
                          batch_size=400).to(device)
    separator.freeze()

    return separator
