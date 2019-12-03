# Performing separation


## Interfacing using the command line

The primary interface to separate files is the command line. To separate a mixture file into the four stems you can just run

```bash
python test.py input_file.wav
```

Note that we support all files that can be read by pysoundfile (wav, flac and ogg files).
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
| `--alpha <float>`         |In case of softmasking, this value changes the exponent to use for building ratio masks. A smaller value usually leads to more interference but better perceptual quality, whereas a larger value leads to less interference but an "overprocessed" sensation.                                                          | `1.0`            |

## Interfacing from python

At the core of the process of separating files is the `separate` function which 
takes a numpy audio array as input (the mixture) and separates into `targets` number of stems.
Note, that for each target a separate model will be loaded and the user would need to know if 
a particular target is available. E.g. for `umx` and `umxhq` the supported targets are 
`['vocals', 'drums', 'bass', 'other']`. The model can be specified using `model_name` parameter.
Both models `umx` and `umxhq` are downloaded automatically. 

The remaining parameters are suggested to set to the default values.

```python
def separate(
    audio,
    targets,
    model_name='umxhq',
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device='cpu'
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
```

