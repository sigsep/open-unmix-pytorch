# Performing separation

## Interfacing using the command line

The primary interface to separate files is the command line. To separate a mixture file into the four stems you can just run

```bash
umx input_file.wav
```

Note that we support all files that can be read by torchaudio, depending on the set backend (either `soundfile` (libsndfile) or `sox`).
For training, we set the default to `soundfile` as it is faster than `sox`. However for inference users might prefer `mp3` decoding capabilities.
The separation can be controlled with additional parameters that influence the performance of the separation.

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
|`--start <float>`  | set start in seconds to reduce the duration of the audio being loaded | `0.0` |
|`--duration <float>`  | set duration in seconds to reduce length of the audio being loaded. Negative values will make the full audio being loaded | `-1.0` |
|`--model <str>`  | path or string of model name to select either a self pre-trained model or a model loaded from `torchhub`.  | |
| `--targets list(str)`           | Targets to be used for separation. For each target a model file with with same name is required.                                                  | `['vocals', 'drums', 'bass', 'other']`          |
| `--niter <int>`           | Number of EM steps for refining initial estimates in a post-processing stage. `--niter 0` skips this step altogether (and thus makes separation significantly faster) More iterations can get better interference reduction at the price of more artifacts.                                                  | `1`          |
| `--residual`           |               computes a residual target, for custom separation scenarios when not all targets are available (at the expense of slightly less performance). E.g vocal/accompaniment can be performed with `--targets vocals --residual`.                                   | not set          |
| `--softmask`       | if activated, then the initial estimates for the sources will be obtained through a ratio mask of the mixture STFT, and not by using the default behavior of reconstructing waveforms by using the mixture phase.  | not set            |
| `--wiener-win-len <int>`         | Number of frames on which to apply filtering independently  | `300`                   |
| `--audio-backend <str>`         | choose audio loading backend, either `sox_io`,  `soundfile` or `stempeg` (which needs additional installation requirements) | [torchaudio default](https://pytorch.org/audio/stable/backend.html) |
| `--aggregate <str>`         | if provided, must be a string containing a valid expression for a dictionary, with keys as output target names, and values a list of targets that are used to build it. For instance: `{ "vocals": ["vocals"], "accompaniment": ["drums", "bass", "other"]}` | `None` |
| `--filterbank <str>`         | filterbank implementation method. Supported: `['torch', 'asteroid']`. While `torch` is ~30% faster compared to `asteroid` on large FFT sizes such as 4096, asteroids STFT maybe be easier to be exported for deployment. | `torch` |

## Interfacing from python

At the core of the process of separating audio is the `Separator` Module which
takes a numpy audio array or a `torch.Tensor` as input (the mixture) and separates into `targets` stems.
Note, that for each target a separate model will be loaded. E.g. for `umx` and `umxhq` the supported targets are
`['vocals', 'drums', 'bass', 'other']`. The models have to be passed to the separators `target_models` parameter.

Both models `umx`, `umxhq`, `umxl` and `umxse` are downloaded automatically.

Here is an example for constructor for the `Separator` takes the following arguments, with suggested default values:

```python
seperator = openunmix.Separator(
    target_models: dict,
    niter: int = 0,
    softmask: bool = False,
    residual: bool = False,
    sample_rate: float = 44100.0,
    n_fft: int = 4096,
    n_hop: int = 1024,
    nb_channels: int = 2,
    wiener_win_len: Optional[int] = 300,
    filterbank: str = 'torch'
):
```

When passing 

> __Caution__ `training` using the EM algorithm (`niter>0`) is not supported. Only plain post-processing is supported right now for gradient computation. This is because the performance overhead of avoiding all the in-places operations is too large.
