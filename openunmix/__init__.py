"""
![sigsep logo](https://sigsep.github.io/hero.png)
Open-Unmix is a deep neural network reference implementation for music source separation, applicable for researchers, audio engineers and artists. Open-Unmix provides ready-to-use models that allow users to separate pop music into four stems: vocals, drums, bass and the remaining other instruments. The models were pre-trained on the MUSDB18 dataset. See details at apply pre-trained model.

This is the python package API documentation. 
Please checkout [the open-unmix website](https://sigsep.github.io/open-unmix) for more information.
"""
from openunmix import utils
import torch.hub


def umxse_spec(targets=None, device="cpu", pretrained=True):
    target_urls = {
        "speech": "https://zenodo.org/api/files/765b45a3-c70d-48a6-936b-09a7989c349a/speech_f5e0d9f9.pth",
        "noise": "https://zenodo.org/api/files/765b45a3-c70d-48a6-936b-09a7989c349a/noise_04a6fc2d.pth",
    }

    from .model import OpenUnmix

    if targets is None:
        targets = ["speech", "noise"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(rate=16000.0, n_fft=1024, bandwidth=16000)

    # load open unmix models speech enhancement models
    target_models = {}
    for target in targets:
        target_unmix = OpenUnmix(
            nb_bins=1024 // 2 + 1, nb_channels=1, hidden_size=256, max_bin=max_bin
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target], map_location=device
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
    device="cpu",
    pretrained=True,
    filterbank="torch",
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
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    Reference:
        Uhlich, Stefan, & Mitsufuji, Yuki. (2020).
        Open-Unmix for Speech Enhancement (UMX SE).
        Zenodo. http://doi.org/10.5281/zenodo.3786908
    """
    from .model import Separator

    target_models = umxse_spec(targets=targets, device=device, pretrained=pretrained)

    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=1024,
        n_hop=512,
        nb_channels=1,
        sample_rate=16000.0,
        filterbank=filterbank,
    ).to(device)

    return separator


def umxhq_spec(targets=None, device="cpu", pretrained=True):
    from .model import OpenUnmix

    # set urls for weights
    target_urls = {
        "bass": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/bass-8d85a5bd.pth",
        "drums": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/drums-9619578f.pth",
        "other": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/other-b52fbbf7.pth",
        "vocals": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/vocals-b62c91ce.pth",
    }

    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000)

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=512, max_bin=max_bin
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target], map_location=device
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
    device="cpu",
    pretrained=True,
    filterbank="torch",
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
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    from .model import Separator

    target_models = umxhq_spec(targets=targets, device=device, pretrained=pretrained)

    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        sample_rate=44100.0,
        filterbank=filterbank,
    ).to(device)

    return separator


def umx_spec(targets=None, device="cpu", pretrained=True):
    from .model import OpenUnmix

    # set urls for weights
    target_urls = {
        "bass": "https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/bass-646024d3.pth",
        "drums": "https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/drums-5a48008b.pth",
        "other": "https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/other-f8e132cc.pth",
        "vocals": "https://zenodo.org/api/files/d6105b95-8c52-430c-84ce-bd14b803faaf/vocals-c8df74a5.pth",
    }

    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000)

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=512, max_bin=max_bin
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target], map_location=device
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
    device="cpu",
    pretrained=True,
    filterbank="torch",
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
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    """

    from .model import Separator

    target_models = umx_spec(targets=targets, device=device, pretrained=pretrained)
    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        sample_rate=44100.0,
        filterbank=filterbank,
    ).to(device)

    return separator


def umxl_spec(targets=None, device="cpu", pretrained=True):
    from .model import OpenUnmix

    # set urls for weights
    target_urls = {
        "bass": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/bass-2ca1ce51.pth",
        "drums": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/drums-69e0ebd4.pth",
        "other": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/other-c8c5b3e6.pth",
        "vocals": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/vocals-bccbd9aa.pth",
    }

    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000)

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=1024, max_bin=max_bin
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target], map_location=device
            )
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def umxl(
    targets=None,
    residual=False,
    niter=1,
    device="cpu",
    pretrained=True,
    filterbank="torch",
):
    """
    Open Unmix Extra (UMX-L), 2-channel/stereo BLSTM Model trained on a private dataset
    of ~400h of multi-track audio.


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
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    """

    from .model import Separator

    target_models = umxl_spec(targets=targets, device=device, pretrained=pretrained)
    separator = Separator(
        target_models=target_models,
        niter=niter,
        residual=residual,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        sample_rate=44100.0,
        filterbank=filterbank,
    ).to(device)

    return separator
