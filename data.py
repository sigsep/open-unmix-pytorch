import random
from pathlib import Path
import torch
import numpy as np

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import musdb
except ImportError:
    musdb = None


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


class SourcesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        seq_duration=None,
        input_sources_paths=['noises', 'vocals'],
        output_source_path='vocals',
        ext="*.wav",
        sample_rate=44100,
        nb_samples=1000,
        paired=True
    ):
        """A dataset of that assumes folders with unmatched sources
        generating input mixtures matched to a target

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
        # convert sequence duration into samples as torchaudio is samplebased
        if seq_duration is not None:
            self.seq_duration = int(seq_duration * sample_rate)
        else:
            self.seq_duration = None
        self.nb_samples = nb_samples
        self.ext = ext
        self.input_sources_paths = input_sources_paths
        self.output_source_path = output_source_path
        self.sources_samples = self.get_track_paths()

    def __getitem__(self, index):
        input_tuple = random_product(*self.sources_samples)
        sources = list(map(self.load_audio, input_tuple))
        mix = torch.stack(sources, dim=0).sum(dim=0)
        ind = self.input_sources_paths.index(self.output_source_path)
        target = self.load_audio(input_tuple[ind])
        return mix, target

    def __len__(self):
        return self.nb_samples

    def get_samples(self, fp):
        # get length of file in samples
        si, _ = torchaudio.info(str(fp))
        # sox_info state the length as the `nb_samples * nb_channels`
        return si.length // si.channels

    def load_audio(self, fp):
        # loads the full track duration
        if self.seq_duration is None:
            sig, rate = torchaudio.load(fp)
            return sig
        # otherwise loads a random excerpt
        else:
            # compare if length is larger than excerpt length
            nb_samples = self.get_samples(fp)
            seek_pos = random.randint(0, nb_samples - self.seq_duration)
            sig, rate = torchaudio.load(
                fp, num_frames=self.seq_duration, offset=seek_pos
            )
            return sig

    def get_track_paths(self):
        """Loads input and output tracks"""

        sources_paths = []
        for sources_folder in self.input_sources_paths:
            p = Path(self.root, sources_folder)
            sources_paths.append(list(p.glob(self.ext)))

        return sources_paths


class TargetsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path,
        seq_duration=None,
        input_target_file='mixture.wav',
        output_target_file='vocals.wav',
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
        self.root_path = Path(root_path).expanduser()
        self.sample_rate = sample_rate
        # convert sequence duration into samples as torchaudio is samplebased
        if seq_duration is not None:
            self.seq_duration = int(seq_duration * sample_rate)
        # set the input and output files (accept glob)
        self.input_target_file = input_target_file
        self.output_target_file = output_target_file
        self.audio_files = list(self.get_track_paths())

    def __getitem__(self, index):
        input_path, output_path = self.audio_files[index]
        X_audio = self.load_audio(input_path)
        Y_audio = self.load_audio(output_path)
        return X_audio, Y_audio

    def __len__(self):
        return len(self.audio_files)

    def get_samples(self, fp):
        # get length of file in samples
        si, _ = torchaudio.info(str(fp))
        # sox_info state the length as the `nb_samples * nb_channels`
        return si.length // si.channels

    def load_audio(self, fp):
        # loads the full track duration
        if self.seq_duration is None:
            sig, rate = torchaudio.load(fp)
            return sig
        # otherwise loads a random excerpt
        else:
            # compare if length is larger than excerpt length
            nb_samples = self.get_samples(fp)
            seek_pos = random.randint(0, nb_samples - self.seq_duration)
            sig, rate = torchaudio.load(
                fp, num_frames=self.seq_duration, offset=seek_pos
            )
            return sig

    def get_track_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root_path)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                input_path = list(track_folder.glob(self.input_target_file))
                output_path = list(track_folder.glob(self.output_target_file))
                # if both targets are available in the subfolder add them
                if input_path and output_path:
                    yield input_path[0], output_path[0]


class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=None,
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


if __name__ == "__main__":
    # dataset iterator test
    dataset = MUSDBDataset(
        seq_duration=1.0,
        download=True,
        subsets="train",
        validation_split='train'
    )
    print(len(dataset))
    for x, y in dataset:
        print(x.shape)
