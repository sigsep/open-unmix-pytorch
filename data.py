import random
from pathlib import Path
import torch
import musdb
import torchaudio


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
                sig, rate = torchaudio.load(
                    fp, num_frames=self.seq_duration, offset=seek_pos
                )
                assert rate == self.sample_rate
                return sig

    @property
    def get_total_duration(self):
        total_duration = 0
        for track_paths in self.audio_files:
            total_duration += self.get_duration(track_paths['x'])
        return total_duration

    def _get_track_samples(self):
        samples = [
            range(0, self.get_samples(track_paths['x']) - self.excerpt, self.excerpt) for track_paths in self.audio_files
        ]
        return samples

    def get_track_paths(self):
        p = Path(self.root)
        for track_folder in p.iterdir():
            if track_folder.is_dir():
                input_path = list(track_folder.glob(self.targets['x']))
                output_path = list(track_folder.glob(self.targets['y']))
                # if both targets are available in the subfolder add them
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
        validation_split='train',
        samples_per_track=16,
        *args, **kwargs
    ):
        """MUSDB18 Dataset wrapper
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

    def __getitem__(self, index):
        # get musdb track object
        sample = self.samples[index]
        track = self.mus.tracks[sample['trk']]
        track.start = sample['pos']
        track.dur = self.seq_duration
        x = track.audio.T
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def create_sample_indices(self):
        samples = []
        for index, track in enumerate(self.mus.tracks):
            for n in range(self.samples_per_track):
                start = random.uniform(0, track.duration - self.seq_duration)
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
