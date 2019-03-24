import os.path
import aubio
import numpy as np
from pathlib import Path
import torchaudio
from torch.utils import data
import numpy as np
import random
import torch

class RRDataset(data.Dataset):
    def __init__(
        self,
        root,
        seq_dur=None,
        targets={'x': 'mixture.wav', 'y': 'vocals.wav'},
        sample_rate=44100,
    ):  
        """Random'n'Raw Dataset
        
        Scales to a large amount of audio data.
        Uses pytorch' index based sample access
        """

        self.root = Path(root).expanduser()
        self.sample_rate = sample_rate
        if seq_dur is not None:
            self.seq_len = int(seq_dur * sample_rate)
        else:
            self.seq_len = None
        # set the input and output files (accept glob)
        self.targets = targets
        self.audio_files = list(self.get_track_paths())

    def __getitem__(self, index):
        X_audio = self.load_audio(self.audio_files[index]['x'])
        Y_audio = self.load_audio(self.audio_files[index]['y'])
        return X_audio, Y_audio

    def __len__(self):
        return len(self.audio_files)

    def get_duration(self, fp):
        # get length of file in samples
        si, _ = torchaudio.info(str(fp))
        # sox_info state the length as the `nb_samples * nb_channels`
        return si.length // si.channels / si.rate

    def get_samples(self, fp):
        # get length of file in samples
        si, _ = torchaudio.info(str(fp))
        # sox_info state the length as the `nb_samples * nb_channels`
        return si.length // si.channels

    def load_audio(self, fp):
        if self.seq_len is None:
            sig, rate = torchaudio.load(fp)
            assert rate == self.sample_rate
            return sig
        else:
            # compare if length is larger than excerpt length
            nb_samples = self.get_samples(fp)
            if nb_samples > self.seq_len:
                seek_pos = random.randint(0, nb_samples - self.seq_len)
                sig, rate = torchaudio.load(fp, num_frames=self.seq_len, offset=seek_pos)
                assert rate == self.sample_rate
                return sig

    @property
    def get_total_duration(self):
        total_duration = 0
        for track_paths in self.audio_files:
            total_duration += self.get_duration(track_paths['x'])
        return total_duration

    def _get_track_samples(self):
        samples = [range(0, self.get_samples(track_paths['x']) -
                         self.excerpt, self.excerpt) for track_paths in self.audio_files]
        return samples

    def get_track_paths(self):
        p = Path(self.root)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                input_path = list(track_folder.glob(self.targets['x']))
                output_path = list(track_folder.glob(self.targets['y']))
                # if both targets are available in the subfolder add them to the track array
                if input_path and output_path:
                    yield {'x': input_path[0], 'y': output_path[0]}


class MUSDBDataset(data.Dataset):
    def __init__(
        self,
        root=None,
        is_wav=False,
        subsets=['train'],
        target='vocals',
        seq_duration=None
    ):
        """MUSDB18 Dataset wrapper
        """
        import musdb
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.mus = musdb.DB(root_dir=root, is_wav=is_wav, subsets=subsets)

    def __getitem__(self, index):
        if self.is_wav:
            # get paths and load using torch audio
            pass
        else:
            # get musdb track object
            track = self.mus[index]
            if self.seq_duration is None:
                start = 0
                dur = None
            else:
                start = random.uniform(0, track.duration - self.seq_duration)
                dur = self.seq_duration

            track.start = start
            track.dur = dur
            X = track.audio.T
            Y = track.targets[self.target].audio.T

            return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.mus)


if __name__ == "__main__":
    # dataset iterator test
    dataset =  MUSDBDataset(seq_duration=1.0)
    for X, Y in dataset:
        print(X.shape)