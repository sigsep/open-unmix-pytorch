# Training Open-Unmix

## Datasets

_open-unmix_ uses standard PyTorch [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) classes. The repository comes with __five__ different datasets which cover a wide range of tasks and applications around source separation. Furthermore we also provide a template Dataset if you want to start using your own dataset. The dataset can be selected through a command line argument:

| Argument      | Description                                                            | Default      |
|----------------------------|------------------------------------------------------------------------|--------------|
| `--dataset <str>`          | Name of the dataset (select from `musdb`, `aligned`, `sourcefolder`, `trackfolder_var`, `trackfolder_fix`) | `musdb`      |
| `--root <str>`           | path to root of dataset on disk.                                                  | `None`       |

### `MUSDBDataset` (musdb)

The [MUSDB18](https://sigsep.github.io/datasets/musdb.html) and [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) are the largest freely available datasets for professionally produced music tracks (~10h duration) of different styles. They come with isolated `drums`, `bass`, `vocals` and `others` stems. _MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.

Training `MUSDB18` using _open-unmix_ comes with several design decisions to improve efficiency and performance.

* __chunking__: we don not feed full audio tracks into _open-unmix_ but instead chunk the audio into 6s excerpts (`--seq-dur 6.0`).
* __balanced track sampling__: to not create a bias for longer audio tracks we randomly yield one track from MUSDB18 and select a random chunk subsequently. In one epoch we select (on average) 64 samples from each track.
* __source augmentation__: we apply random gains between `0.25` and `1.25` to all sources before mixing. Furthermore, we randomly swap the channels the input mixture.
* __random track mixing__: for a given target we select a _random track_ with replacement. To yield a mixture we draw the interfering sources from different tracks (again with replacement) to increase generalization of the model.
* __fixed validation split__: we provide a fixed validation split of [16 tracks](https://github.com/sigsep/sigsep-mus-db/blob/b283da5b8f24e84172a60a06bb8f3dacd57aa6cd/musdb/configs/mus.yaml#L41). We evaluate on these tracks in full length instead of using chunking to have evaluation as close as possible to the actual test data.

#### Dataset arguments

| Argument      | Description                                                            | Default      |
|---------------------|-----------------------------------------------|--------------|
| `--is-wav`          | loads the decoded WAVs instead of STEMS for faster data loading. See [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional). | `musdb`      |
| `--samples-per-track <int>` | sets the number of samples that are randomly drawn from each track  | `64`       |
| `--source-augmentations <list[str]>` | applies augmentations to each audio source before mixing | `gain channelswap`       |

Note that, if `--root` is not specified, we automatically download a 7 second preview version of the MUSDB18 dataset. While this is comfortable for testing purposes, we wouldn't recommend to actually train your model on this.

#### Example

To train the vocal model with _open-unmix_ using the MUSDB18 dataset use the following arguments:

```bash
python train.py --dataset musdb --root /data/musdb --target vocals
```

### `AlignedDataset` (aligned)

This dataset assumes multiple track folders where each track includes one single input and one output file, directly corresponding to the input and the output of the model.

This dataset is the most basic of all datasets provided here, due to the least amount of
preprocessing, it is also the fastest option, however, it lacks any kind of source augmentations or custom mixing. Instead, it directly uses the target files that are within the folder. The filenames would have to be identical for each track. E.g, for the first sample of the training, input could be `1/mixture.wav` and output could be `1/vocals.wav`.

Typical use cases:

* Source Separation (Mixture -> Target)
* Denoising (Noisy -> Clean)
* Bandwidth Extension (Low Bandwidth -> High Bandwidth)

#### File Structure

```
data/train/01/mixture.wav --> input
data/train/01/vocals.wav ---> output
```

#### Example

```bash
python train.py --dataset aligned --root /data/data --input_file mixture.wav --output_file vocals.wav
```

### `SourceFolderDataset` (sourcefolder)

A dataset of that assumes folders of sources,
instead of track folders. This is a common
format for speech and environmental sound datasets
such das DCASE. For each source a variable number of
tracks/sounds is available, therefore the dataset is unaligned by design.

In this scenario one could easily train a network to separate a target sounds from interfering sounds. For each sample, the data loader loads a random combination of target+interferer as the input and performs a linear mixture of these. The output of the model is the target.

#### File structure

```
train/vocals/track11.wav -----------------\
train/drums/track202.wav  (interferer1) ---+--> input
train/bass/track007a.wav  (interferer2) --/

train/vocals/track11.wav ---------------------> output
```

#### Parameters

...

#### Example

```bash
python train.py --dataset sourcefolder --root /data --target-dir vocals --interferer-dirs carnoise windnoise --ext .ogg --nb-train-samples 1000
```

### `FixedSourcesTrackFolderDataset` (trackfolder_fix)

A dataset of that assumes audio sources to be stored
in track folder where each track has a fixed number of sources. For each track the users specifies the target file-name (`target_file`) and a list of interferences files (`interferer_files`).
A linear mix is performed on the fly by summing the target and the interferers up.

Due to the fact that all tracks comprise the exact same set of sources, the random track mixing augmentation technique can be used, where sources from different tracks are mixed together. Setting `random_track_mix=True` results in an unaligned dataset.
When random track mixing is enabled, we define an epoch as when the the target source from all tracks has been seen and only once with whatever interfering sources has randomly been drawn.

This dataset is recommended to be used for small/medium size for example like the MUSDB18 or other custom source separation datasets.

#### File structure

```sh
train/1/vocals.wav ---------------\
train/1/drums.wav (interferer1) ---+--> input
train/1/bass.wav -(interferer2) --/

train/1/vocals.wav -------------------> output
```

#### Example

```
python train.py  --root /data --dataset trackfolder_fix --target-file vocals.flac --interferer-files bass.flac drums.flac other.flac
```

### `VariableSourcesTrackFolderDataset` (trackfolder_var)

A dataset of that assumes audio sources to be stored in track folder where each track has a _variable_ number of sources. The users specifies the target file-name (`target_file`) and the extension of sources to used for mixing. A linear mix is performed on the fly by summing all sources in a track folder.

Since the number of sources differ per track, while target is fixed, a random track mix augmentation cannot be used.

Also make sure, that you do not provide the mixture file among the sources!

#### File structure

```sh
train/1/vocals.wav --> input target   \
train/1/drums.wav --> input target     |
train/1/bass.wav --> input target    --+--> input
train/1/accordion.wav --> input target |
train/1/marimba.wav --> input target  /

train/1/vocals.wav -----------------------> output
```

#### Example

```
python train.py --root /data --dataset trackfolder_var --target-file vocals.flac --ext .wav
```

### Template Dataset

```python
from utils import load_audio, load_info
class TemplateDataset(torch.utils.data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, root, split='train', target='vocals'):
        """Initialize the dataset
        """
        self.root = root
        self.tracks = get_tracks(root, split)

    def __getitem__(self, index):
        """Return a single training example
        """
        path = self.tracks[index]
        x = load_audio(path)
        y = load_audio(path)
        return x, y

    def __len__(self):
        """Return the number of audio samples"""
        return len(self.tracks)
```

### Template Model

```python
from model import Spectrogram, STFT, NoOp
class Model(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        input_is_spectrogram=False,
        sample_rate=44100,
    ):
        """
        Input:  (batch, channel, sample)
            or  (frame, batch, channels, frequency)
        Output: (frame, batch, channels, frequency)
        """

        super(OpenUnmix, self).__init__()
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)


    def forward(self, mix):
        # transform to spectrogram on the fly
        X = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # transform X to estimate
        # ....

        return X
```

## Training and Model Parameters

Additional training parameters and their default values are listed below:

| Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--output <str>`           | path where to save the trained output model as well as checkpoints.                         | `./umx`      |
| `--no_cuda`           | disable cuda even if available                                              | not set      |
| `--epochs <int>`           | number of epochs to train                                                       | `1000`          |
| `--patience <int>`         | early stopping patience                                                         | `250`            |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `16`            |
| `--seq-dur <int>`          | Sequence duration in seconds of excerpts taken from the dataset.                | `6.0`           |
| `--lr <float>`             | learning rate                                                                   | `0.0001`        |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--nfft <int>`             | STFT FFT window length in samples                                               | `4096`          |
| `--nhop <int>`             | STFT hop length in samples                                                      | `1024`          |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--bandwidth <int>`        | maximum bandwidth in Hertz processed by the LSTM. Output is always full bandwidth!                                                | `16000`         |
| `--nb-channels <int>`      | set number of channels for model (1 for mono, 2 for stereo)                     | `2`             |
| `--quiet`                  | disable print and progress bar during training                                   | not set         |

## Output Files

* `args`: All command line parameters that were used to train the model
* `

### Training details of `umxhq` and `umx`

* parameters for both models are identical
* around 80 seconds per epoch on an Nvidia RTX2080.
* we ran 4 different seeds and for each target collected the model with the lowest validation loss.

![umx-hq](https://user-images.githubusercontent.com/72940/61230598-9e6e3b00-a72a-11e9-8a89-aca1862341eb.png)


### Arguments