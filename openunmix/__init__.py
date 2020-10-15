from openunmix import utils
import torch.hub


def umxse_spec(targets=None, device='cpu', pretrained=True):
    target_urls = {
        'speech': 'https://zenodo.org/api/files/765b45a3-c70d-48a6-936b-09a7989c349a/speech_f5e0d9f9.pth',
        'noise': 'https://zenodo.org/api/files/765b45a3-c70d-48a6-936b-09a7989c349a/noise_04a6fc2d.pth'
    }

    from . model import OpenUnmix

    if targets is None:
        targets = ['speech', 'noise']

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(
        rate=16000.0,
        n_fft=1024,
        bandwidth=16000
    )

    # load open unmix models speech enhancement models
    target_models = {}
    for target in targets:
        target_unmix = OpenUnmix(
            nb_bins=1024 // 2 + 1,
            nb_channels=1,
            hidden_size=256,
            max_bin=max_bin
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target],
                map_location=device
            )
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def umxse(
    targets=None,
    residual=False,
    niter=1,
    device='cpu',
    pretrained=True
):
    """
    Open Unmix Speech Enhancemennt 1-channel BiLSTM Model
    trained on the 28-speaker version of Voicebank+Demand
    (Sampling rate: 16kHz)

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: ['speech', 'noise'].
                If you don't pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference

    Reference:
        Uhlich, Stefan, & Mitsufuji, Yuki. (2020).
        Open-Unmix for Speech Enhancement (UMX SE).
        Zenodo. http://doi.org/10.5281/zenodo.3786908
    """
    from . model import Separator

    target_models = umxse_spec(
        targets=targets,
        device=device,
        pretrained=pretrained
    )

    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=1024,
        n_hop=512,
        nb_channels=1,
        sample_rate=16000.0
    ).to(device)

    return separator


def umxhq_spec(
    targets=None,
    device='cpu',
    pretrained=True
):
    from . model import OpenUnmix

    # set urls for weights
    target_urls = {
        'bass': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/bass-8d85a5bd.pth',
        'drums': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/drums-9619578f.pth',
        'other': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/other-b52fbbf7.pth',
        'vocals': 'https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/vocals-b62c91ce.pth'
    }

    if targets is None:
        targets = ['vocals', 'drums', 'bass', 'other']

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(
        rate=44100.0,
        n_fft=4096,
        bandwidth=16000
    )

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1,
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
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def umxhq(
    targets=None,
    residual=False,
    niter=1,
    device='cpu',
    pretrained=True
):
    """
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18-HQ

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: ['vocals', 'drums', 'bass', 'other'].
                If you don't pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference
    """

    from . model import Separator

    target_models = umxhq_spec(
        targets=targets,
        device=device,
        pretrained=pretrained
    )
    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        sample_rate=44100.0
    ).to(device)

    return separator


def umx_spec(
    targets=None,
    device='cpu',
    pretrained=True
):
    from . model import OpenUnmix

    # set urls for weights
    target_urls = {
        'bass': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/bass-646024d3.pth',
        'drums': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/drums-5a48008b.pth',
        'other': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/other-f8e132cc.pth',
        'vocals': 'https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/vocals-c8df74a5.pth'
    }

    if targets is None:
        targets = ['vocals', 'drums', 'bass', 'other']

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(
        rate=44100.0,
        n_fft=4096,
        bandwidth=16000
    )

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1,
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
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def umx(
    targets=None,
    residual=False,
    niter=1,
    device='cpu',
    pretrained=True,
):
    """
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: ['vocals', 'drums', 'bass', 'other'].
                If you don't pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference

    """

    from . model import Separator

    target_models = umx_spec(
        targets=targets,
        device=device,
        pretrained=pretrained
    )
    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        sample_rate=44100.0
    ).to(device)

    return separator


def separate(
    input,
    rate=None,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None
):
    """
    Open Unmix functional interface

    Separates a torch.Tensor or the content of an audio file.

    If a separator is provided, use it for inference. If not, create one
    and use it afterwards.


    Args:
        input: audio to process
            * if it's a str: path of the audio to load
            * if it's a torch Tensor: shape (channels, length), and
              `rate` must also be provided.
        rate: int or None: only used if input is a Tensor. Otherwise, 
            inferred from the file.
        model_str_or_path: the pretrained model to use
        targets (str): select the targets for the source to be separated.
                a list including: ['vocals', 'drums', 'bass', 'other'].
                If you don't pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        niter (int): the number of post-processingiterations, defaults to 1
        residual (bool): if True, a "garbage" target is created
        wiener_win_len (int): the number of frames to use when batching
                the post-processing step
        aggregate_dict (str): if provided, must be a string containing a '
             'valid expression for a dictionary, with keys as output '
             'target names, and values a list of targets that are used to '
             'build it. For instance: \'{\"vocals\":[\"vocals\"], '
             '\"accompaniment\":[\"drums\",\"bass\",\"other\"]}\'
        separator: if provided, the model.Separator object that will be used
             to perform separation
        device (str): selects device to be used for inference

    """
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True
        )
        separator.freeze()
        if device:
            separator.to(device)

    if isinstance(input, str):
        # assuming input is a filepath
        import torchaudio
        audio, rate = torchaudio.load(input)
    elif isinstance(input, torch.Tensor):
        audio = input
        if rate is None:
            raise Exception('When `unmix` is fed with a Tensor, `rate` must be provided.')

    if device:
        audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(
        estimates,
        aggregate_dict=aggregate_dict
    )
    return estimates
