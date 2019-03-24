import random
import numpy as np
from pathlib import Path
import torch

class SourceFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        seq_duration=None,
        input_file='mixture.wav',
        output_file='vocals.wav',
        sample_rate=44100,
    ):  
        """Random'n'Raw Dataset
        
        Scales to a large amount of audio data.
        Uses pytorch' index based sample access
        """
        import torchaudio
        self.root = Path(root).expanduser()
        self.sample_rate = sample_rate
        if seq_duration is not None:
            self.seq_duration = int(seq_duration * sample_rate)
        # set the input and output files (accept glob)
        self.targets = {'x': input_file, 'y': output_file}
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
        if self.seq_duration is None:
            sig, rate = torchaudio.load(fp)
            assert rate == self.sample_rate
            return sig
        else:
            # compare if length is larger than excerpt length
            nb_samples = self.get_samples(fp)
            if nb_samples > self.seq_duration:
                seek_pos = random.randint(0, nb_samples - self.seq_duration)
                sig, rate = torchaudio.load(fp, num_frames=self.seq_duration, offset=seek_pos)
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


class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=None,
        is_wav=False,
        subsets=['train'],
        target='vocals',
        seq_duration=None,
        download=True,
        validation_split='train',
        *args, **kwargs
    ):
        """MUSDB18 Dataset wrapper
        """
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.validation_split = validation_split
        self.mus = musdb.DB(
            root_dir=root, download=download, is_wav=is_wav, validation_split=validation_split, subsets=subsets, *args, **kwargs
        )

    def __getitem__(self, index):
        # get musdb track object
        track = self.mus[index]
        if self.seq_duration is None or self.validation_split != 'train':
            start = 0
            dur = None
        else:
            start = random.uniform(0, track.duration - self.seq_duration)
            dur = self.seq_duration

        track.start = start
        track.dur = dur
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.mus)


if __name__ == "__main__":
    # dataset iterator test
    import sampling
    dataset =  MUSDBDataset(
        seq_duration=1.0, download=True, subsets="train", validation_split='train'
    )
    print(len(dataset))
    for x, y in dataset:
        print(x.shape)