#  _Open-Unmix_ for PyTorch

[![status](https://joss.theoj.org/papers/571753bc54c5d6dd36382c3d801de41d/status.svg)](https://joss.theoj.org/papers/571753bc54c5d6dd36382c3d801de41d) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mijF0zGWxN-KaxTnd0q6hayAlrID5fEQ)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-unmix-a-reference-implementation-for/music-source-separation-on-musdb18)](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=open-unmix-a-reference-implementation-for)

[![Build Status](https://github.com/sigsep/open-unmix-pytorch/workflows/CI/badge.svg)](https://github.com/sigsep/open-unmix-pytorch/actions?query=workflow%3ACI+branch%3Amaster+event%3Apush)
[![Latest Version](https://img.shields.io/pypi/v/openunmix.svg)](https://pypi.python.org/pypi/openunmix)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/openunmix.svg)](https://pypi.python.org/pypi/openunmix)

This repository contains the PyTorch (1.8+) implementation of __Open-Unmix__, a deep neural network reference implementation for music source separation, applicable for researchers, audio engineers and artists. __Open-Unmix__ provides ready-to-use models that allow users to separate pop music into four stems: __vocals__, __drums__, __bass__ and the remaining __other__ instruments. The models were pre-trained on the freely available [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. See details at [apply pre-trained model](#getting-started).

## ⭐️ News 

- 03/07/2021: We added `umxl`, a model that was trained on extra data which significantly improves the performance, especially generalization.
- 14/02/2021: We released the new version of open-unmix as a python package. This comes with: a fully differentiable version of [norbert](https://github.com/sigsep/norbert), improved audio loading pipeline and large number of bug fixes. See [release notes](https://github.com/sigsep/open-unmix-pytorch/releases/) for further info.

- 06/05/2020: We added a pre-trained speech enhancement model `umxse` provided by Sony.

- 13/03/2020: Open-unmix was awarded 2nd place in the [PyTorch Global Summer Hackathon 2020](https://devpost.com/software/open-unmix).

__Related Projects:__ open-unmix-pytorch | [open-unmix-nnabla](https://github.com/sigsep/open-unmix-nnabla) | [musdb](https://github.com/sigsep/sigsep-mus-db) | [museval](https://github.com/sigsep/sigsep-mus-eval) | [norbert](https://github.com/sigsep/norbert)

## 🧠 The Model (for one source)

![](https://docs.google.com/drawings/d/e/2PACX-1vTPoQiPwmdfET4pZhue1RvG7oEUJz7eUeQvCu6vzYeKRwHl6by4RRTnphImSKM0k5KXw9rZ1iIFnpGW/pub?w=959&h=308)

To perform separation into multiple sources, _Open-unmix_ comprises multiple models that are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.

Each _Open-Unmix_ source model is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target source, like _vocals_, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error.

### Input Stage

__Open-Unmix__ operates in the time-frequency domain to perform its prediction. The input of the model is either:

* __`models.Separator`:__ A time domain signal tensor of shape `(nb_samples, nb_channels, nb_timesteps)`, where `nb_samples` are the samples in a batch, `nb_channels` is 1 or 2 for mono or stereo audio, respectively, and `nb_timesteps` is the number of audio samples in the recording. In this case, the model computes STFTs with either `torch` or `asteroid_filteranks` on the fly.

* __`models.OpenUnmix`:__ The core open-unmix takes **magnitude spectrograms** directly (e.g. when pre-computed and loaded from disk). In that case, the input is of shape `(nb_frames, nb_samples, nb_channels, nb_bins)`, where `nb_frames` and `nb_bins` are the time and frequency-dimensions of a Short-Time-Fourier-Transform.

The input spectrogram is _standardized_ using the global mean and standard deviation for every frequency bin across all frames. Furthermore, we apply batch normalization in multiple stages of the model to make the training more robust against gain variation.

### Dimensionality reduction

The LSTM is not operating on the original input spectrogram resolution. Instead, in the first step after the normalization, the network learns to compresses the frequency and channel axis of the model to reduce redundancy and make the model converge faster.

### Bidirectional-LSTM

The core of __open-unmix__ is a three layer bidirectional [LSTM network](https://dl.acm.org/citation.cfm?id=1246450). Due to its recurrent nature, the model can be trained and evaluated on arbitrary length of audio signals. Since the model takes information from past and future simultaneously, the model cannot be used in an online/real-time manner.
An uni-directional model can easily be trained as described [here](docs/training.md).

### Output Stage

After applying the LSTM, the signal is decoded back to its original input dimensionality. In the last steps the output is multiplied with the input magnitude spectrogram, so that the models is asked to learn a mask.

## 🤹‍♀️ Putting source models together: the `Separator`

`models.Separator` puts together _Open-unmix_ spectrogram model for each desired target, and combines their output through a multichannel generalized Wiener filter, before application of inverse STFTs using `torchaudio`.
The filtering is differentiable (but parameter-free) version of [norbert](https://github.com/sigsep/norbert). The separator is currently currently only used during inference.

## 🏁 Getting started

### Installation

`openunmix` can be installed from pypi using:

```
pip install openunmix
```

Note, that the pypi version of openunmix uses [torchaudio] to load and save audio files. To increase the number of supported input and output file formats (such as STEMS export), please additionally install [stempeg](https://github.com/faroit/stempeg).

Training is not part of the open-unmix package, please follow [docs/train.md] for more information.

#### Using Docker

We also provide a docker container. Performing separation of a local track in `~/Music/track1.wav` can be performed in a single line:

```
docker run -v ~/Music/:/data -it faroit/open-unmix-pytorch "/data/track1.wav" --outdir /data/track1
```

### Pre-trained models

We provide three core pre-trained music separation models. All three models are end-to-end models that take waveform inputs and output the separated waveforms.

* __`umxl` (default)__  trained on private stems dataset of compressed stems. __Note, that the weights are only licensed for non-commercial use (CC BY-NC-SA 4.0).__

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5069601.svg)](https://doi.org/10.5281/zenodo.5069601)

* __`umxhq`__  trained on [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#uncompressed-wav) which comprises the same tracks as in MUSDB18 but un-compressed which yield in a full bandwidth of 22050 Hz.

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3370489.svg)](https://doi.org/10.5281/zenodo.3370489)

* __`umx`__ is trained on the regular [MUSDB18](https://sigsep.github.io/datasets/musdb.html#compressed-stems) which is bandwidth limited to 16 kHz do to AAC compression. This model should be used for comparison with other (older) methods for evaluation in [SiSEC18](sisec18.unmix.app).

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3370486.svg)](https://doi.org/10.5281/zenodo.3370486)

Furthermore, we provide a model for speech enhancement trained by [Sony Corporation](link)

* __`umxse`__ speech enhancement model is trained on the 28-speaker version of the [Voicebank+DEMAND corpus](https://datashare.is.ed.ac.uk/handle/10283/1942?show=full).

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3786908.svg)](https://doi.org/10.5281/zenodo.3786908)

All four models are also available as spectrogram (core) models, which take magnitude spectrogram inputs and ouput separated spectrograms.
These models can be loaded using `umxl_spec`, `umxhq_spec`, `umx_spec` and `umxse_spec`.

To separate audio files (`wav`, `flac`, `ogg` - but not `mp3`) files just run:

```bash
umx input_file.wav
```

A more detailed list of the parameters used for the separation is given in the [inference.md](/docs/inference.md) document.

We provide a [jupyter notebook on google colab](https://colab.research.google.com/drive/1mijF0zGWxN-KaxTnd0q6hayAlrID5fEQ) to experiment with open-unmix and to separate files online without any installation setup.

### Using pre-trained models from within python

We implementes several ways to load pre-trained models and use them from within your python projects:
#### When the package is installed

Loading a pre-trained models is as simple as loading

```python
separator = openunmix.umxl(...)
```
#### torch.hub

We also provide a torch.hub compatible modules that can be loaded. Note that this does _not_ even require to install the open-unmix packagen and should generally work when the pytorch version is the same.

```python
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl, device=device)
```

Where, `umxl` specifies the pre-trained model. 
#### Performing separation

With a created separator object, one can perform separation of some `audio` (torch.Tensor of shape `(channels, length)`, provided as at a sampling rate `separator.sample_rate`) through:

```python
estimates = separator(audio, ...)
# returns estimates as tensor
```

Note that this requires the audio to be in the right shape and sampling rate. For convenience we provide a pre-processing in `openunmix.utils.preprocess(..`)` that takes numpy audio and converts it to be used for open-unmix.

#### One-liner

To perform model loading, preprocessing and separation in one step, just use:

```python
from openunmix import separate
estimates = separate.predict(audio, ...)
```

### Load user-trained models

When a path instead of a model-name is provided to `--model`, pre-trained `Separator` will be loaded from disk.
E.g. The following files are assumed to present when loading `--model mymodel --targets vocals`

* `mymodel/separator.json`
* `mymodel/vocals.pth`
* `mymodel/vocals.json`


Note that the separator usually joins multiple models for each target and performs separation using all models. E.g. if the separator contains `vocals` and `drums` models, two output files are generated, unless the `--residual` option is selected, in which case an additional source will be produced, containing an estimate of all that is not the targets in the mixtures.

### Evaluation using `museval`

To perform evaluation in comparison to other SISEC systems, you would need to install the `museval` package using

```
pip install museval
```

and then run the evaluation using

`python -m openunmix.evaluate --outdir /path/to/musdb/estimates --evaldir /path/to/museval/results`

### Results compared to SiSEC 2018 (SDR/Vocals)

Open-Unmix yields state-of-the-art results compared to participants from [SiSEC 2018](https://sisec18.unmix.app/#/methods). The performance of `UMXHQ` and `UMX` is almost identical since it was evaluated on compressed STEMS.

![boxplot_updated](https://user-images.githubusercontent.com/72940/63944652-3f624c80-ca72-11e9-8d33-bed701679fe6.png)

Note that

1. [`STL1`, `TAK2`, `TAK3`, `TAU1`, `UHL3`, `UMXHQ`] were omitted as they were _not_ trained on only _MUSDB18_.
2. [`HEL1`, `TAK1`, `UHL1`, `UHL2`] are not open-source.

#### Scores (Median of frames, Median of tracks)

|target|SDR  | SDR |  SDR |
|------|-----|-----|-----|
|`model`|UMX  |UMXHQ|UMXL |
|vocals|6.32 | 6.25|__7.21__ |
|bass  |5.23 | 5.07|__6.02__ |
|drums |5.73 | 6.04|__7.15__ |
|other |4.02 | 4.28|__4.89__ |

## Training

Details on the training is provided in a separate document [here](docs/training.md).

## Extensions

Details on how _open-unmix_ can be extended or improved for future research on music separation is described in a separate document [here](docs/extensions.md).


## Design Choices

we favored simplicity over performance to promote clearness of the code. The rationale is to have __open-unmix__ serve as a __baseline__ for future research while performance still meets current state-of-the-art (See [Evaluation](#Evaluation)). The results are comparable/better to those of `UHL1`/`UHL2` which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app).
We designed the code to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. We favored framework specifics implementations instead of having a monolithic repository with common code for all frameworks.

## How to contribute

_open-unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and requests for technical support through [github issues](https://github.com/sigsep/open-unmix-pytorch/issues/new/choose). For more details of how to contribute, please follow our [`CONTRIBUTING.md`](CONTRIBUTING.md). For help and support, please use the gitter chat or the google groups forums. 

### Authors

[Fabian-Robert Stöter](https://www.faroit.com/), [Antoine Liutkus](https://github.com/aliutkus), Inria and LIRMM, Montpellier, France

## References

<details><summary>If you use open-unmix for your research – Cite Open-Unmix</summary>

```latex
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-Unmix - A Reference Implementation for Music Source Separation},  
  journal={Journal of Open Source Software},  
  year=2019,
  doi = {10.21105/joss.01667},
  url = {https://doi.org/10.21105/joss.01667}
}
```

</p>
</details>

<details><summary>If you use the MUSDB dataset for your research - Cite the MUSDB18 Dataset</summary>
<p>

```latex
@misc{MUSDB18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```

</p>
</details>


<details><summary>If compare your results with SiSEC 2018 Participants - Cite the SiSEC 2018 LVA/ICA Paper</summary>
<p>

```latex
@inproceedings{SiSEC18,
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",
  title="The 2018 Signal Separation Evaluation Campaign",
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",
  year="2018",
  pages="293--305"
}
```

</p>
</details>

⚠️ Please note that the official acronym for _open-unmix_ is **UMX**.

### License

MIT

### Acknowledgements

<p align="center">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/logo_INRIA.svg?sanitize=true" width="200" title="inria">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/anr.jpg" width="100" alt="anr">
</p>
