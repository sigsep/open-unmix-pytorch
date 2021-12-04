# Training Open-Unmix

> This documentation refers to the standard training procedure for _Open-unmix_, where each target is trained independently. It has not been updated for the end-to-end training capabilities that the `Separator` module allows. Please contribute if you try this.

Both models, `umxhq` and `umx` that are provided with pre-trained weights, can be trained using the default parameters of the `scripts/train.py` function.

## Installation

The train function is not part of the python package, thus we suggest to use [Anaconda](https://anaconda.org/) to install the training requirments, also because the environment would allow reproducible results.

To create a conda environment for _open-unmix_, simply run:

`conda env create -f scripts/environment-X.yml` where `X` is either [`cpu-linux`, `gpu-linux-cuda10`, `cpu-osx`], depending on your system. For now, we haven't tested windows support.

## Training API

The [MUSDB18](https://sigsep.github.io/datasets/musdb.html) and [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) are the largest freely available datasets for professionally produced music tracks (~10h duration) of different styles. They come with isolated `drums`, `bass`, `vocals` and `others` stems. _MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.

To directly train a vocal model with _open-unmix_, we first would need to download one of the datasets and place in _unzipped_ in a directory of your choice (called `root`).

| Argument | Description | Default |
|----------|-------------|---------|
| `--root <str>` | path to root of dataset on disk.                                                  | `None`       |

Also note that, if `--root` is not specified, we automatically download a 7 second preview version of the MUSDB18 dataset. While this is comfortable for testing purposes, we wouldn't recommend to actually train your model on this.

Training can be started using

```bash
python train.py --root path/to/musdb18 --target vocals
```

Training `MUSDB18` using _open-unmix_ comes with several design decisions that we made as part of our defaults to improve efficiency and performance:

* __chunking__: we do not feed full audio tracks into _open-unmix_ but instead chunk the audio into 6s excerpts (`--seq-dur 6.0`).
* __balanced track sampling__: to not create a bias for longer audio tracks we randomly yield one track from MUSDB18 and select a random chunk subsequently. In one epoch we select (on average) 64 samples from each track.
* __source augmentation__: we apply random gains between `0.25` and `1.25` to all sources before mixing. Furthermore, we randomly swap the channels the input mixture.
* __random track mixing__: for a given target we select a _random track_ with replacement. To yield a mixture we draw the interfering sources from different tracks (again with replacement) to increase generalization of the model.
* __fixed validation split__: we provide a fixed validation split of [14 tracks](https://github.com/sigsep/sigsep-mus-db/blob/b283da5b8f24e84172a60a06bb8f3dacd57aa6cd/musdb/configs/mus.yaml#L41). We evaluate on these tracks in full length instead of using chunking to have evaluation as close as possible to the actual test data.

Some of the parameters for the MUSDB sampling can be controlled using the following arguments:

| Argument      | Description                                                            | Default      |
|---------------------|-----------------------------------------------|--------------|
| `--is-wav`          | loads the decoded WAVs instead of STEMS for faster data loading. See [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional). | `True`      |
| `--samples-per-track <int>` | sets the number of samples that are randomly drawn from each track  | `64`       |
| `--source-augmentations <list[str]>` | applies augmentations to each audio source before mixing, available augmentations: `[gain, channelswap]`| [gain, channelswap]       |

## Training and Model Parameters

An extensive list of additional training parameters allows researchers to quickly try out different parameterizations such as a different FFT size. The table below, we list the additional training parameters and their default values (used for `umxhq` and `umx`L:

| Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--target <str>`           | name of target source (will be passed to the dataset)                         | `vocals`      |
| `--output <str>`           | path where to save the trained output model as well as checkpoints.                         | `./open-unmix`      |
| `--checkpoint <str>`           | path to checkpoint of target model to resume training. | not set      |
| `--model <str>`           | path or str to pretrained target to fine-tune model | not set      |
| `--no_cuda`           | disable cuda even if available                                              | not set      |
| `--epochs <int>`           | Number of epochs to train                                                       | `1000`          |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `16`            |
| `--patience <int>`         | early stopping patience                                                         | `140`            |
| `--seq-dur <int>`          | Sequence duration in seconds of chunks taken from the dataset. A value of `<=0.0` results in full/variable length               | `6.0`           |
| `--unidirectional`           | changes the bidirectional LSTM to unidirectional (for real-time applications)  | not set      |
| `--hidden-size <int>`             | Hidden size parameter of dense bottleneck layers  | `512`            |
| `--nfft <int>`             | STFT FFT window length in samples                                               | `4096`          |
| `--nhop <int>`             | STFT hop length in samples                                                      | `1024`          |
| `--lr <float>`             | learning rate                                                                   | `0.001`        |
| `--lr-decay-patience <int>`             | learning rate decay patience for plateau scheduler                                                                   | `80`        |
| `--lr-decay-gamma <float>`             | gamma of learning rate plateau scheduler.  | `0.3`        |
| `--weight-decay <float>`             | weight decay for regularization                                                                   | `0.00001`        |
| `--bandwidth <int>`        | maximum bandwidth in Hertz processed by the LSTM. Input and Output is always full bandwidth! | `16000`         |
| `--nb-channels <int>`      | set number of channels for model (1 for mono (spectral downmix is applied,) 2 for stereo)                     | `2`             |
| `--nb-workers <int>`      | Number of (parallel) workers for data-loader, can be safely increased for wav files   | `0` |
| `--quiet`                  | disable print and progress bar during training                                   | not set         |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--audio-backend <str>`         | choose audio loading backend, either `sox` or `soundfile` | `soundfile` for training, `sox` for inference |

### Training details of `umxhq`

The training of `umxhq` took place on Nvidia RTX2080 cards. Equipped with fast SSDs and `--nb-workers 4`, we could utilize around 90% of the GPU, thus training time was around 80 seconds per epoch. We ran four different seeds for each target and selected the model with the lowest validation loss.

The training and validation loss curves are plotted below:

![umx-hq](https://user-images.githubusercontent.com/72940/61230598-9e6e3b00-a72a-11e9-8a89-aca1862341eb.png)

## Other Datasets

_open-unmix_ uses standard PyTorch [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) classes. The repository comes with __five__ different datasets which cover a wide range of tasks and applications around source separation. Furthermore we also provide a template Dataset if you want to start using your own dataset. The dataset can be selected through a command line argument:

| Argument      | Description                                                            | Default      |
|----------------------------|------------------------------------------------------------------------|--------------|
| `--dataset <str>`          | Name of the dataset (select from `musdb`, `aligned`, `sourcefolder`, `trackfolder_var`, `trackfolder_fix`) | `musdb`      |

### `AlignedDataset` (aligned)

This dataset assumes multiple track folders, where each track includes an input and one output file, directly corresponding to the input and the output of the model.

This dataset is the most basic of all datasets provided here, due to the least amount of
preprocessing, it is also the fastest option, however, it lacks any kind of source augmentations or custom mixing. Instead, it directly uses the target files that are within the folder. The filenames would have to be identical for each track. E.g, for the first sample of the training, input could be `1/mixture.wav` and output could be `1/vocals.wav`.

Typical use cases:

* Source Separation (Mixture -> Target)
* Denoising (Noisy -> Clean)
* Bandwidth Extension (Low Bandwidth -> High Bandwidth)

#### File Structure

```
data/train/1/mixture.wav --> input
data/train/1/vocals.wav ---> output
...
data/valid/1/mixture.wav --> input
data/valid/1/vocals.wav ---> output

```

#### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
|`--input-file <str>` | input file name | `None` |
|`--output-file <str>` | output file name | `None` |

#### Example

```bash
python train.py --dataset aligned --root /dataset --input_file mixture.wav --output_file vocals.wav
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

| Argument | Description | Default |
|----------|-------------|---------|
|`--interferer-dirs list[<str>]` | list of directories used as interferers | `None` |
|`--target-dir <str>` | directory that contains the target source | `None` |
|`--ext <str>` | File extension | `.wav` |
|`--ext <str>` | File extension | `.wav` |
|`--nb-train-samples <str>` | Number of samples drawn for training | `1000` |
|`--nb-valid-samples <str>` | Number of samples drawn for validation | `100` |
|`--source-augmentations list[<str>]` | List of augmentation functions that are processed in the order of the list | |

#### Example

```bash
python train.py --dataset sourcefolder --root /data --target-dir vocals --interferer-dirs car_noise wind_noise --ext .ogg --nb-train-samples 1000
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

#### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
|`--target-file <str>` | Target file (includes extension) | `None` |
|`--interferer-files list[<str>]` | list of interfering sources | `None` |
|`--random-track-mix` | Applies random track mixing | `False` |
|`--source-augmentations list[<str>]` | List of augmentation functions that are processed in the order of the list | |

#### Example

```
python train.py  --root /data --dataset trackfolder_fix --target-file vocals.flac --interferer-files bass.flac drums.flac other.flac
```

### `VariableSourcesTrackFolderDataset` (trackfolder_var)

A dataset of that assumes audio sources to be stored in track folder where each track has a _variable_ number of sources. The users specifies the target file-name (`target_file`) and the extension of sources to used for mixing. A linear mix is performed on the fly by summing all sources in a track folder.

Since the number of sources differ per track, while target is fixed, a random track mix augmentation cannot be used.
Also make sure, that you do not provide the mixture file among the sources! This dataset maximizes the number of tracks that can be used since it doesn't require the presence of a fixed number of sources per track. However, it is required to
have the target file to be present. To increase the dataset utilization even further users can enable the `--silence-missing-targets` option that outputs silence to missing targets.

#### File structure

```sh
train/1/vocals.wav --> input target   \
train/1/drums.wav --> input target     |
train/1/bass.wav --> input target    --+--> input
train/1/accordion.wav --> input target |
train/1/marimba.wav --> input target  /

train/1/vocals.wav -----------------------> output
```

#### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
|`--target-file <str>` | file name of target file | `None` |
|`--silence-missing-targets` | if a target is not among the list of sources it will be filled with zero | not set |
|`random interferer mixing` | use _random track_ for the inference track to increase generalization of the model. | not set |
|`--ext <str>` | File extension that is used to find the interfering files | `.wav` |
|`--source-augmentations list[<str>]` | List of augmentation functions that are processed in the order of the list | |

#### Example

```
python train.py --root /data --dataset trackfolder_var --target-file vocals.flac --ext .wav
```
