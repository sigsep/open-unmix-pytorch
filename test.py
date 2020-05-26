import torch
import argparse
import soundfile as sf
from pathlib import Path
from filtering import Separator
import utils


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
        '--residual',
        type=str,
        default=None,
        help='if provided, build a source with given name for the mix '\
             'minus all estimated targets'
    )

    inf_parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='if provided, must be a string containing a valid expression for '
             'a dictionary, with keys the output target names, and values '
             'a list of targets that are used to build it. For instance: '
             '\'{\"vocals\":[\"vocals\"], \'accompaniment\':[\"drums\",'
             '\"bass\",\"other\"]}\''
    )
    return inf_parser.parse_args()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='UMX Inference',
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

    torch.autograd.set_detect_anomaly(True)

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    gradient = False
    print('testing gradients:', gradient)
    if gradient:
        torch.autograd.set_detect_anomaly(True)

    # create the Separator object
    separator = Separator(targets=args.targets,
                          model_name=args.model,
                          niter=args.niter,
                          softmask=args.softmask,
                          alpha=args.alpha,
                          residual=args.residual,
                          out=args.out,
                          device=device,
                          batch_size=100, preload=True)
    separator.freeze()

    # loop over the files
    for input_file in args.input:
        # handling an input audio path
        audio, rate = sf.read(input_file, always_2d=True)

        # convert numpy audio to torch
        audio_torch = torch.tensor(audio).to(device)
        audio_torch = utils.as_stereo_batch(audio)

        audio_torch.requires_grad = gradient

        # getting the separated signals
        estimates, model_rate = separator(audio_torch, rate)

        if gradient:
            loss = torch.abs(estimates['vocals']).sum()
            loss.backward()

        estimates = {
            key: estimates[key][0].detach().cpu().numpy()
            for key in estimates}

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
                model_rate
            )
