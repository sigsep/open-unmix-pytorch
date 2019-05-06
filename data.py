import random
from pathlib import Path
import torch
import torch.utils.data
import numpy as np
import sys


try:
    import soundfile as sf
except ImportError:
    soundfile = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import musdb
except ImportError:
    musdb = None


def soundfile_info(path):
    info = {}
    sfi = sf.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return info


def soundfile_loader(path, start=0, dur=None):
    # get metadata
    info = soundfile_info(path)
    start = int(start * info['samplerate'])
    # check if dur is none
    if dur:
        # stop in soundfile is calc in samples, not seconds
        stop = start + int(dur * info['samplerate'])
    else:
        # set to None for reading complete file
        stop = dur

    audio, _ = sf.read(
        path,
        always_2d=True,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def torchaudio_info(path):
    # get length of file in samples
    info = {}
    si, _ = torchaudio.info(str(path))
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels
    info['duration'] = info['samples'] / si.rate
    return info


def torchaudio_loader(path, start=0, dur=None):
    info = torchaudio_info(path)
    # loads the full track duration
    if dur is None:
        sig, rate = torchaudio.load(path)
        return sig
        # otherwise loads a random excerpt
    else:
        num_frames = int(dur * info['samplerate'])
        offset = int(start * info['samplerate'])
        sig, rate = torchaudio.load(
            path, num_frames=num_frames, offset=offset
        )
        return sig


def audioloader(path, start=0, dur=None):
    if 'torchaudio' in sys.modules:
        return torchaudio_loader(path, start=start, dur=dur)
    else:
        return soundfile_loader(path, start=start, dur=dur)


def audioinfo(path):
    if 'torchaudio' in sys.modules:
        return torchaudio_info(path)
    else:
        return soundfile_info(path)


def load_datasets(parser, args):
    if args.dataset == 'unaligned':
        parser.add_argument('--valid_split', type=float, default="0.1")
        parser.add_argument('--interferences', type=str, nargs="+")
        args = parser.parse_args()

        sources_dataset = UnalignedSources(
            root=Path(args.root),
            seq_duration=args.seq_dur,
            target=args.target,
            interferences=args.interferences
        )

        valid_split = args.valid_split

        lengths = [
            len(sources_dataset) - int(len(sources_dataset)*valid_split),
            int(len(sources_dataset)*valid_split)
        ]
        train_dataset, valid_dataset = torch.utils.data.random_split(
            sources_dataset, lengths
        )
        train_dataset.sample_rate = sources_dataset.sample_rate
    elif args.dataset == 'aligned':
        parser.add_argument('--valid_split', type=float, default="0.1")
        parser.add_argument('--input_file', type=str)
        parser.add_argument('--output_file', type=str)

        args = parser.parse_args()
        # set output target to basename of output file
        args.target = Path(args.output_file).stem
        sources_dataset = AlignedSources(
            root=Path(args.root),
            seq_duration=args.seq_dur,
            input_file=args.input_file,
            output_file=args.output_file,
        )

        valid_split = args.valid_split

        lengths = [
            len(sources_dataset) - int(len(sources_dataset)*valid_split),
            int(len(sources_dataset)*valid_split)
        ]
        train_dataset, valid_dataset = torch.utils.data.random_split(
            sources_dataset, lengths
        )
        train_dataset.sample_rate = sources_dataset.sample_rate

    elif args.dataset == 'musdb':
        parser.add_argument('--is-wav', action='store_true', default=False,
                            help='flags wav version of the dataset')

        args = parser.parse_args()
        dataset_kwargs = {
            'root': args.root,
            'is_wav': args.is_wav,
            'seq_duration': args.seq_dur,
            'subsets': 'train',
            'target': args.target,
            'download': args.root is None
        }

        train_dataset = MUSDBDataset(
            validation_split='train', **dataset_kwargs
        )
        valid_dataset = MUSDBDataset(
            validation_split='valid', **dataset_kwargs
        )

    return train_dataset, valid_dataset, args


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


class UnalignedSources(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        seq_duration=None,
        target='drums',
        interferences=['noise'],
        glob="*.wav",
        sample_rate=44100,
        nb_samples=1000,
    ):
        """A dataset of that assumes sources to be unaligned,
        organized in subfolders with the name of sources

        Example:
            -- Sample 1 ----------------------

            /noise/10923.wav --+
                               +--> mixed input
            /vocals/1.wav -----+
            /vocals/1.wav --------> output target

        Scales to a large amount of audio data.
        Uses pytorch' index based sample access
        """
        self.root = Path(root).expanduser()
        self.sample_rate = sample_rate
        if seq_duration <= 0:
            self.seq_duration = None
        else:
            self.seq_duration = seq_duration
        self.nb_samples = nb_samples
        self.glob = glob
        self.source_folders = interferences + [target]
        self.sources = self._get_paths()

    def __getitem__(self, index):
        input_tuple = random_product(*self.sources)
        sample_sources = list(map(self.load_audio, input_tuple))
        mix = torch.stack(sample_sources, dim=0).sum(dim=0)
        target = self.load_audio(input_tuple[-1])
        return mix, target

    def __len__(self):
        return self.nb_samples

    def load_audio(self, fp):
        # loads the full track duration
        return audioloader(fp, start=0, dur=self.seq_duration)

    def _get_paths(self):
        """Loads input and output tracks"""

        sources_paths = []
        for source_folder in self.source_folders:
            p = Path(self.root, source_folder)
            sources_paths.append(list(p.glob(self.glob)))

        return sources_paths


class AlignedSources(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        input_file='mixture.wav',
        output_file='vocals.wav',
        seq_duration=None,
        sample_rate=44100,
    ):
        """A dataset of that assumes folders with sources

        Example:
            -- Sample 1 ----------------------
            /01/mixture.wav --> input target
            /01/vocals.wav ---> output target
            -- Sample 2 -----------------------
            /02/mixture.wav --> input target
            /02/vocals.wav ---> output target

        Scales to a large amount of audio data.
        Uses pytorch' index based sample access
        """
        self.root = Path(root).expanduser()
        self.sample_rate = sample_rate
        if seq_duration <= 0:
            self.seq_duration = None
        else:
            self.seq_duration = seq_duration
        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())

    def __getitem__(self, index):
        input_path, output_path = self.tuple_paths[index]
        X_audio = self.load_audio(input_path)
        Y_audio = self.load_audio(output_path)
        return X_audio, Y_audio

    def __len__(self):
        return len(self.tuple_paths)

    def load_audio(self, fp):
        return audioloader(fp, start=0, dur=self.seq_duration)

    def _get_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                input_path = list(track_folder.glob(self.input_file))
                output_path = list(track_folder.glob(self.output_file))
                # if both targets are available in the subfolder add them
                if input_path and output_path:
                    yield input_path[0], output_path[0]


class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=None,
        download=False,
        is_wav=False,
        subsets=['train'],
        target='vocals',
        seq_duration=None,
        validation_split='train',
        samples_per_track=64,
        dtype=torch.float32,
        *args, **kwargs
    ):
        """MUSDB18 Dataset wrapper that samples from the musdb tracks
        using excerpts without replacement spaced
        """
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.validation_split = validation_split
        self.samples_per_track = samples_per_track
        self.mus = musdb.DB(
            root_dir=root,
            is_wav=is_wav,
            validation_split=validation_split,
            subsets=subsets,
            download=download,
            *args, **kwargs
        )
        self.samples = self.create_sample_indices()
        self.sample_rate = 44100
        self.dtype = dtype

    def __getitem__(self, index):
        # get musdb track object
        sample = self.samples[index]
        track = self.mus.tracks[sample['trk']]
        track.start = sample['pos']
        track.dur = self.seq_duration
        x = torch.tensor(track.audio.T, dtype=self.dtype)
        y = torch.tensor(track.targets[self.target].audio.T, dtype=self.dtype)
        return x, y

    def __len__(self):
        return len(self.samples)

    def create_sample_indices(self):
        samples = []
        for index, track in enumerate(self.mus.tracks):
            # compute a fixed number of segements per track
            sample_positions = np.linspace(
                0, track.duration - self.seq_duration,
                self.samples_per_track
            )
            for start in sample_positions:
                samples.append({
                    'trk': index,
                    'pos': start
                })
        if self.validation_split == 'train':
            random.seed(42)
            random.shuffle(samples)
        return samples
