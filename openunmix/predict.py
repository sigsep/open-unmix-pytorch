import argparse

import torchaudio

from openunmix import utils


def inference_args(parser):
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
        '--wiener-win-len',
        type=int,
        default=300,
        help='Number of frames on which to apply filtering independently'
    )

    inf_parser.add_argument(
        '--residual',
        type=str,
        default=None,
        help='if provided, build a source with given name'
             'for the mix minus all estimated targets'
    )

    inf_parser.add_argument(
        '--aggregate',
        type=str,
        default=None,
        help='if provided, must be a string containing a valid expression for '
             'a dictionary, with keys as output target names, and values '
             'a list of targets that are used to build it. For instance: '
             '\'{\"vocals\":[\"vocals\"], \"accompaniment\":[\"drums\",'
             '\"bass\",\"other\"]}\''
    )
    return inf_parser.parse_args()


def unmix(
    input_file,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None
):
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

    # loop over the files
    # handling an input audio path
    audio, rate = torchaudio.load(input_file)
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
