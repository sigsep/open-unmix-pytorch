import torch
import argparse
import soundfile as sf
from pathlib import Path
import model
from model import Separator
import utils
import json


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--residual',
        type=str,
        default=None,
        help='if provided, build a source with given name'
             'for the mix minus all estimated targets'
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
    targets = model.load_models(
        targets=args.targets,
        model_name=args.model
    )

    # parsing the output dict
    out = None if args.out is None else json.loads(args.out)

    separator = Separator(targets=targets,
                          niter=args.niter,
                          residual=args.residual,
                          out=out,
                          wiener_win_len=300).to(device)
    separator.freeze()

    # loop over the files
    for input_file in args.input:
        # handling an input audio path
        audio, rate = sf.read(input_file, always_2d=True)

        # convert numpy audio to torch
        audio_torch = torch.as_tensor(audio).to(device)
        audio_torch = utils.preprocess(audio, rate, separator.sample_rate)

        audio_torch.requires_grad = gradient

        # getting the separated signals
        estimates = separator(audio_torch)

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
                separator.sample_rate
            )
