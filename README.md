# _Open-Unmix_ for PyTorch

![](https://sisec18.unmix.app/static/img/hero_header.4f28952.svg)

This repository contains the PyTorch (1.0+) implementation of __open-unmix__, a deep neural network reference implementation for music source separation, applicable for researchers, audio engineers and artists.

__open-unmix__ provides ready-to-use pre-trained models that allow users to separate pop music into four stems: __vocals__, __drums__, __bass__ and the remaining __other__ instruments.

We provide models that were pre-trained  on the [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. See details at [apply pre-trained model](#inference).

We also provide implementations for [tensorflow]() and [nnabla]().

## The Model

_Open-Unmix_ is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target, like _vocals_, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error and the actual separation is done in a post-processing step involving a multichannel wiener filter implemented using [norbert](https://github.com/sigsep/norbert). To perform separation into multiple sources, multiple models are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.

### Input Stage

_open-unmix_ operates in the time-frequency domain to perform its prediction. The input of the model is either:

* __A time domain__ signal tensor of shape `(nb_samples, nb_channels, nb_timesteps)`, where `nb_samples` are the samples in a batch, `nb_channels` is 1 or 2 for mono or stereo audio, respectively, and `nb_timesteps` is the number of audio samples in the recording.

 In that case, the model computes spectrograms with `torch.STFT` on the fly.

* Alternatively _open-unmix_ also takes **magnitude spectrograms** directly (e.g. when pre-computed and loaded from disk).

 In that case, the input is of shape `(nb_frames, nb_samples, nb_channels, nb_bins)`, where `nb_frames` and `nb_bins` are the time and frequency-dimensions of a Short-Time-Fourier-Transform.

The input spectrogram is _standardized_ using the global mean and standard deviation for every frequency bin across all frames. Furthermore, we apply batch normalization in multiple stages of the model to make the training more robust against gain variation.

### Dimensionality reduction

The LSTM is not operating on the original input spectrogram resolution. Instead, in the first step after the normalization, the network learns to compresses the frequency and channel axis of the model to reduce redundancy and make the model converge faster.

### Bidirectional-LSTM

The core of __open-unmix__ is a three layer bidirectional [LSTM network](https://dl.acm.org/citation.cfm?id=1246450). Due to its recurrent nature, the model can be trained and evaluated on arbitrary length of audio signals. Since the model takes information from past and future simultaneously, the model cannot be used in an online/real-time manner.
An uni-directional model can easily be trained as described [here](docs/training.md).

### Output Stage

After applying the LSTM, the signal is decoded back to its original input dimensionality. In the last steps the output is multiplied with the input magnitude spectrogram, so that the models is asked to learn a mask.

## Separation

Since PyTorch currently lacks an invertible STFT, the synthesis is performed in numpy. For inference, we rely on [an implementation](https://github.com/sigsep/norbert) of a multichannel Wiener filter that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation. The `norbert` module assumes to have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture.

## Getting started

### Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu-linux`, `gpu-cuda10`, `cpu-osx`], depending on your system. For now, we haven't tested windows support.

### Applying the pre-trained model on audio files

To separate audio files (wav, flac, ogg) files just run:

```bash
python test.py input_file.wav
```

Additionally `--model umx` can be used to load a different pre-trained models, we currently support the following:

* __`umxhq` (default)__ is trained on [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#MUSDB18HQ) which comprises the same tracks as in MUSDB18 but un-compressed which yield in a full bandwidth of 22050 Hz.

* __`umx`__ is trained on the regular [MUSDB18](https://sigsep.github.io/datasets/musdb.html) which is bandlimited to 16 kHz do to AAC compression. This model should be used for comparison with other (older) methods for evaluation in [SiSEC18](sisec18.unmix.app).

### Separation Parameters

The separation can be controlled with additional parameters that influence the performance of the separation

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--targets list(str)`           | Targets to be used for separation. For each target a model file with with same name is required.                                                  | `['vocals', 'drums', 'bass', 'other']`          |
| `--softmask`       | if activated, then the initial estimates for the sources will be obtained through a ratio mask of the mixture STFT, and not by using the default behavior of reconstructing waveforms by using the mixture phase.  | not set            |
| `--niter <int>`           | Number of EM steps for refining initial estimates in a post-processing stage. `--niter 0` skips this step altogether. More iterations can get better interference reduction at the price of more artifacts.                                                  | `1`          |
| `--alpha <float>`         |In case of softmasking, this value changes the exponent to use for building ratio masks. A smaller value usually leads to more interference but better perceptual quality, whereas a larger value leads to less interference but an "overprocessed" sensation.                                                          | `1.0`            |

### Jupyter Notebook

We provide a [notebook on google colab](https://colab.research.google.com/drive/1mijF0zGWxN-KaxTnd0q6hayAlrID5fEQ) to experiment with open-unmix and to separate files online without any installation setup.

### Load user models

```bash
python test.py --model /path/to/model/root/directory input_file.wav
```

Note that `model` usually contains individual models for each target and performs separation using all models. E.g. if `model_path` contains `vocals` and `drums` models, two output files are generated.

### Evaluation using `museval`

To perform evaluation in comparison to other SISEC systems, you would need to install the `museval` package using

```
pip install museval
```

and then run the evaluation using

`python eval.py --outdir /path/to/musdb/estimates --evaldir /path/to/museval/results`

### Results compared to SiSEC 2018

Open-Unmix yields state-of-the-art results compared to participants from [SiSEC 2018](https://sisec18.unmix.app). The performance of `UMXHQ` and `UMX` is almost identical since it was evaluated on compressed STEMS. e 

![boxplot](https://user-images.githubusercontent.com/72940/61356654-16497c00-a877-11e9-967c-2d2ab5e0801c.png)

Also note that

* [`STL1`, `STL2`, `TAK2`, `TAK3`, `TAU1`, `UHL3`] used additional training datasets.
* [`HEL1`, `RPCA`, `TAK1`, `TAK2`, `TAK3`, `TAU1`, `UHL1`, `UHL2`, `UHL3`] do not provide any cod

## Training

See separate document [here](docs/training.md).

## Design Choices / Contributions

* we favored simplicity over performance to promote clearness of the code. The rationale is to have __open-unmix__ serve as a __baseline__ for future research while performance still meets current state-of-the-art (See [Evaluation](#Evaluation)). The results are comparable/better to those of `UHL1`/`UHL2` which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app).
* We designed the code to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. We favored framework specifics implementations instead of having a monolithic repository.
* _open-unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and comments and improve the computational performance. However, we are not looking for changes that only focused on improving the performance.

### Authors

[Fabian-Robert Stöter](https://www.faroit.com/), [Antoine Liutkus](https://github.com/aliutkus), Inria and LIRMM, Montpellier, France

## References

<details><summary>If you use open-unmix for your research – Cite Open-Unmix</summary>
  
```latex
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-unmix: a reference implementation for source separation},  
  journal={Journal of Open-Source Research},  
  year=2019,  
  note={submitted}}
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
