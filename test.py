import torch
import numpy as np
import argparse
import soundfile as sf
from torchaudio.functional import istft
import json
from pathlib import Path
import scipy.signal
import resampy
import model
import utils
import warnings
import tqdm
from contextlib import redirect_stderr
import filtering
import io


def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            state['sample_rate'],
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model.OpenUnmix(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def separate(
    audio,
    targets,
    model_name='umxhq',
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu'
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device
        )
        Vj = unmix_target(audio_torch)
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ..., None])  # remove sample dim and add source dim
        source_names += [target]
    # Creating a Tensor out of the list:
    # (nb_frames, nb_channels, nb_bins, nb_sources)
    V = torch.cat(V, dim=-1)

    # transposing it as (nb_frames, nb_bins, {1,nb_channels}, nb_sources)
    V = V.permute(0, 2, 1, 3).detach().cpu().to(torch.float64)

    # getting the STFT of mix: (nb_samples, nb_channels, nb_bins, nb_frames, 2)
    X = unmix_target.stft(audio_torch).detach().cpu().to(torch.float64)

    # rearranging it into: (nb_frames, nb_bins, nb_channels, 2) to feed into
    # filtering methods
    X = X[0].permute(2, 1, 0, 3)

    if residual_model or len(targets) == 1:
        V = filtering.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    # initializing the result
    nb_sources = V.shape[-1]
    nb_frames = V.shape[0]
    Y = torch.zeros(X.shape + (nb_sources, ), dtype=torch.float32,
                    device=X.device)
    for t in torch.utils.data.DataLoader(torch.arange(nb_frames),
                                         batch_size=300):
        Y[t] = filtering.wiener(V[t], X[t], niter, use_softmask=softmask
                                ).to(torch.float32)

    estimates = {}
    # getting to (channel, fft_size, n_frames, 2, nb_sources)
    Y = Y.permute(2, 1, 0, 3, 4)

    for j, name in enumerate(source_names):
        estimates[name] = istft(
            Y[..., j],
            n_fft=unmix_target.stft.n_fft,
            hop_length=unmix_target.stft.n_hop,
            window=unmix_target.stft.window.cpu(),
            center=unmix_target.stft.center,
            normalized=False, onesided=True,
            pad_mode='reflect', length=audio_torch.shape[-1]
        ).T
    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in args.input:
        # handling an input audio path
        print(input_file)
        audio, rate = sf.read(input_file, always_2d=True)

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != args.samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, args.samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            targets=args.targets,
            model_name=args.model,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device
        )
        # import ipdb; ipdb.set_trace()
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + '_' + args.model)
            else:
                outdir = Path(Path(input_file).stem + '_' + model_path.stem)
        else:
            outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        for target, estimate in estimates.items():
            sf.write(
                outdir / Path(target).with_suffix('.wav'),
                estimate,
                args.samplerate
            )
