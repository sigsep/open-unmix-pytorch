from pathlib import Path
import torch
import torchaudio
import json
import numpy as np
import tqdm

from openunmix import utils
from openunmix import predict
from openunmix import data

import argparse


def separate():
    parser = argparse.ArgumentParser(
        description="UMX Inference",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=str, nargs="+", help="List of paths to wav/flac files.")

    parser.add_argument(
        "--model",
        default="umxl",
        type=str,
        help="path to mode base directory of pretrained models, defaults to UMX-L",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        help="provide targets to be processed. \
              If none, all available targets will be computed",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Output extension which sets the audio format",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    parser.add_argument(
        "--duration",
        type=float,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    parser.add_argument(
        "--audio-backend",
        type=str,
        help="Sets audio backend. Default to torchaudio's default backend: See https://pytorch.org/audio/stable/backend.html"
        "(`sox_io`, `sox`, `soundfile` or `stempeg`)",
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=1,
        help="number of iterations for refining results.",
    )

    parser.add_argument(
        "--wiener-win-len",
        type=int,
        default=300,
        help="Number of frames on which to apply filtering independently",
    )

    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="if provided, build a source with given name "
        "for the mix minus all estimated targets",
    )

    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    parser.add_argument(
        "--filterbank",
        type=str,
        default="torch",
        help="filterbank implementation method. "
        "Supported: `['torch', 'asteroid']`. `torch` is ~30%% faster "
        "compared to `asteroid` on large FFT sizes such as 4096. However "
        "asteroids stft can be exported to onnx, which makes is practical "
        "for deployment.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable log messages",
    )
    args = parser.parse_args()

    if args.audio_backend != "stempeg" and args.audio_backend is not None:
        torchaudio.set_audio_backend(args.audio_backend)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.verbose:
        print("Using ", device)
    # parsing the output dict
    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    # create separator only once to reduce model loading
    # when using multiple files
    separator = utils.load_separator(
        model_str_or_path=args.model,
        targets=args.targets,
        niter=args.niter,
        residual=args.residual,
        wiener_win_len=args.wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=args.filterbank,
    )

    separator.freeze()
    separator.to(device)

    if args.audio_backend == "stempeg":
        try:
            import stempeg
        except ImportError:
            raise RuntimeError("Please install pip package `stempeg`")

    # loop over the files
    for input_file in tqdm.tqdm(args.input):
        if args.audio_backend == "stempeg":
            audio, rate = stempeg.read_stems(
                input_file,
                start=args.start,
                duration=args.duration,
                sample_rate=separator.sample_rate,
                dtype=np.float32,
            )
            audio = torch.tensor(audio)
        else:
            audio, rate = data.load_audio(input_file, start=args.start, dur=args.duration)
        estimates = predict.separate(
            audio=audio,
            rate=rate,
            aggregate_dict=aggregate_dict,
            separator=separator,
            device=device,
        )
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + "_" + args.model)
            else:
                outdir = Path(Path(input_file).stem + "_" + model_path.stem)
        else:
            outdir = Path(args.outdir) / Path(input_file).stem
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        if args.audio_backend == "stempeg":
            target_path = str(outdir / Path("target").with_suffix(args.ext))
            # convert torch dict to numpy dict
            estimates_numpy = {}
            for target, estimate in estimates.items():
                estimates_numpy[target] = torch.squeeze(estimate).detach().cpu().numpy().T

            stempeg.write_stems(
                target_path,
                estimates_numpy,
                sample_rate=separator.sample_rate,
                writer=stempeg.FilesWriter(multiprocess=True, output_sample_rate=rate),
            )
        else:
            for target, estimate in estimates.items():
                target_path = str(outdir / Path(target).with_suffix(args.ext))
                torchaudio.save(
                    target_path,
                    torch.squeeze(estimate).to("cpu"),
                    sample_rate=separator.sample_rate,
                )
