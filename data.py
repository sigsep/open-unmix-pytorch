from utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
import numpy as np
import argparse
import random
import musdb
import torch
import tqdm

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def augment_source_gain(audio, low=0.25, high=1.25):
    g = low + torch.rand(1) * (high - low)
    return audio * g


def augment_source_channelswap(audio):
    if audio.shape[0] == 2 and random.random() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(parser, args):
    if args.dataset == 'unaligned':
        parser.add_argument('--interfer_files', type=str, nargs="+")
        args = parser.parse_args()

        dataset_kwargs = {
            'root': Path(args.root),
            'seq_duration': args.seq_dur,
            'target': args.target,
            'interfer_files': args.interfer_files
        }

        train_dataset = UnalignedSources(split='train', **dataset_kwargs)
        valid_dataset = UnalignedSources(split='valid', **dataset_kwargs)

    elif args.dataset == 'aligned':
        parser.add_argument('--input-file', type=str)
        parser.add_argument('--output-file', type=str)

        args = parser.parse_args()
        # set output target to basename of output file
        args.target = Path(args.output_file).stem

        dataset_kwargs = {
            'root': Path(args.root),
            'seq_duration': args.seq_dur,
            'input_file': args.input_file,
            'output_file': args.output_file
        }

        train_dataset = AlignedDataset(split='train', **dataset_kwargs)
        valid_dataset = AlignedDataset(split='valid', **dataset_kwargs)

    elif args.dataset == 'sources':
        parser.add_argument('--interfer-files', type=str, nargs='+')
        parser.add_argument('--target-file', type=str)

        args = parser.parse_args()

        dataset_kwargs = {
            'root': Path(args.root),
            'interfer_files': args.interfer_files,
            'target_file': args.target_file
        }

        source_augmentations = Compose(
            [augment_source_channelswap, augment_source_gain]
        )

        train_dataset = SourcesDataset(
            split='train',
            source_augmentations=source_augmentations,
            random_track_mix=True,
            random_chunk=True,
            seq_duration=args.seq_dur,
            **dataset_kwargs
        )
        valid_dataset = SourcesDataset(
            split='valid', 
            seq_duration=-1,
            **dataset_kwargs
        )

    elif args.dataset == 'musdb':
        parser.add_argument('--is-wav', action='store_true', default=False,
                            help='flags wav version of the dataset')
        parser.add_argument('--samples-per-track', type=int, default=64)

        args = parser.parse_args()
        dataset_kwargs = {
            'root': args.root,
            'is_wav': args.is_wav,
            'subsets': 'train',
            'target': args.target,
            'download': args.root is None
        }

        source_augmentations = Compose(
            [augment_source_channelswap, augment_source_gain]
        )

        train_dataset = MUSDBDataset(
            split='train',
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            **dataset_kwargs
        )

        valid_dataset = MUSDBDataset(
            split='valid', samples_per_track=1, seq_duration=None,
            **dataset_kwargs
        )

    return train_dataset, valid_dataset, args


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


class UnalignedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        seq_duration=None,
        target_dir='drums',
        interferences_dirs=['noise'],
        glob="*.wav",
        sample_rate=44100,
        nb_samples=1000,
    ):
        """A dataset of that assumes sources to be unaligned,
        organized in subfolders with the name of sources

        Example:
            -- Sample 1 ----------------------

            train/noise/10923.wav --+
                                    +--> mixed input
            train/vocals/1.wav -----+
            train/vocals/1.wav --------> output target

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
        sample_sources = list(map(load_audio, input_tuple))
        mix = torch.stack(sample_sources, dim=0).sum(dim=0)
        target = load_audio(input_tuple[-1], start=0, dur=self.seq_duration)
        return mix, target

    def __len__(self):
        return self.nb_samples

    def _get_paths(self):
        """Loads input and output tracks"""

        sources_paths = []
        for source_folder in self.source_folders:
            p = Path(self.root, self.split, source_folder)
            sources_paths.append(list(p.glob(self.glob)))

        return sources_paths


class AlignedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        input_file='mixture.wav',
        output_file='vocals.wav',
        seq_duration=None,
        random_chunk=False,
        sample_rate=44100,
    ):
        """A dataset of that assumes folders with sources

        Example:
            -- Sample 1 ----------------------
            train/01/mixture.wav --> input target
            train/01/vocals.wav ---> output target
            -- Sample 2 -----------------------
            train/02/mixture.wav --> input target
            train/02/vocals.wav ---> output target

        Scales to a large amount of audio data.
        Uses pytorch' index based sample access
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        if seq_duration <= 0:
            self.seq_duration = None
        else:
            self.seq_duration = seq_duration
        self.random_chunk = random_chunk
        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())

    def __getitem__(self, index):
        input_path, output_path = self.tuple_paths[index]
        input_info = load_info(input_path)
        output_info = load_info(output_path)
        if self.random_chunk:
            # use the minimum of x and y in case they differ
            duration = min(input_info['duration'], output_info['duration'])
            if duration < self.seq_duration:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            # random start in seconds
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0
        try:
            X_audio = load_audio(
                input_path, start=start, dur=self.seq_duration
            )
            Y_audio = load_audio(
                output_path, start=start, dur=self.seq_duration
            )
        except RuntimeError:
            print("error in ", input_path, output_path)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

        if X_audio.shape[1] < int(self.seq_duration * input_info['samplerate']) or Y_audio.shape[1] < int(self.seq_duration * output_info['samplerate']):
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        return X_audio, Y_audio

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                input_path = list(track_folder.glob(self.input_file))
                output_path = list(track_folder.glob(self.output_file))
                # if both targets are available in the subfolder add them
                if input_path and output_path:
                    yield input_path[0], output_path[0]


class SourcesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        target_file='vocals.wav',
        interfer_files=['bass.wav', 'drums.wav'],
        seq_duration=None,
        random_chunk=False,
        random_track_mix=False,
        sample_rate=44100,
        source_augmentations=lambda audio: audio,
    ):
        """A dataset of that assumes folders with sources

        Example:
            -- Sample 1 ----------------------
            train/01/vocals.wav --> input target   \
            train/01/drums.wav --> input target  --+--> will be mixed
            train/01/bass.wav --> input target    /
            train/01/vocals.wav ---> output target
            -- Sample 2 -----------------------
            train/02/vocals.wav --> input target   \
            train/02/drums.wav --> input target  --+--> will be mixed
            train/02/bass.wav --> input target    /
            train/02/vocals.wav ---> output target

        Scales to a large amount of audio data.
        Uses pytorch' index based sampling
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        if seq_duration <= 0:
            self.seq_duration = None
        else:
            self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunk = random_chunk
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interfer_files = interfer_files
        self.tracks = list(self.get_tracks())

    def __getitem__(self, index):
        audio_sources = []
        sources = self.interfer_files + [self.target_file]
        for source in sources:
            # select a random track for each source
            if self.random_track_mix:
                track_dir = random.choice(self.tracks)
            else:
                track_dir = self.tracks[index]
            source_path = track_dir / source
            if source_path.exists():
                input_info = load_info(source_path)
                duration = input_info['duration']

            if self.random_chunk:
                start = random.uniform(0, duration - self.seq_duration)
            else:
                start = 0

            audio = load_audio(
                source_path, start=start, dur=self.seq_duration
            )
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the last element in the list
        y = stems[-1]
        return x, y

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                yield track_folder


class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target='vocals',
        root=None,
        download=False,
        is_wav=False,
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=torch.float32,
        seed=42,
        *args, **kwargs
    ):
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples that are yielded from each musdb track
            in one epoch. Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        dtype : numeric type
            data type of torch output tuple x and y
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args, **kwargs
        )
        self.sample_rate = 44100  # musdb is fixed sample rate
        self.dtype = dtype

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train':
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(
                    0, track.duration - self.seq_duration
                )
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(
                    track.sources[source].audio.T,
                    dtype=self.dtype
                )
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype=self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype=self.dtype
            )

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')
    parser.add_argument(
        '--dataset', type=str, default="musdb",
        choices=['musdb', 'aligned', 'unaligned', 'sources'],
        help='Name of the dataset.'
    )

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help=('write out a fixed dataset of samples')
    )

    parser.add_argument('--target', type=str, default='vocals')

    # I/O Parameters
    parser.add_argument(
        '--seq-dur', type=float, default=5.0,
        help='Duration of <=0.0 will result in the full audio'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    train_dataset, valid_dataset, args = load_datasets(parser, args)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    if args.save:
        for k, (x, y) in enumerate(train_dataset):
            torchaudio.save(
                "test/" + str(k) + 'x.wav',
                x,
                44100,
                precision=16,
                channels_first=True
            )
            torchaudio.save(
                "test/" + str(k) + 'y.wav',
                y,
                44100,
                precision=16,
                channels_first=True
            )

    # check datasampler
    for x, y in tqdm.tqdm(train_sampler):
        pass
