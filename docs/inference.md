# Performing separation

## Interfacing using the command line

The primary interface to separate files is the command line. To separate a mixture file into the four stems you can just run

```bash
python test.py input_file.wav
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
| `--alpha <float>`         |In case of softmasking, this value changes the exponent to use for building ratio masks. A smaller value usually leads to more interference but better perceptual quality, whereas a larger value leads to less interference but an "overprocessed" sensation.                                                          | `1.0`                   |
| `--audio-backend <str>`         | choose audio loading backend, either `sox` or `soundfile` | `soundfile` for training, `sox` for inference |

## Interfacing from python

At the core of the process of separating audio is the `Separator` Module which
takes a numpy audio array or a torch.Tensor as input (the mixture) and separates into `targets` stems.
Note, that for each target a separate model will be loaded and the user would need to know if
a particular target is available. E.g. for `umx` and `umxhq` the supported targets are
`['vocals', 'drums', 'bass', 'other']`. The model can be specified using `model_str_or_path` parameter.
Both models `umx` and `umxhq` are downloaded automatically.

the constructor for the Separator class takes the following arguments, with suggested default values:

```python
class Separator(nn.Module):
  def __init__(
    target_models,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    batch_size=None,
    training=False,
    device='cpu',
    smart_input_management='True'
    )
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Parameters
    ----------
    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

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

    batch_size: {None | int}
        The size of the batches (number of frames) on which to apply filtering
        independently. This means assuming time varying stereo models and
        localization of sources.
        None means not batching but using the whole signal. It comes at the
        price of a much larger memory usage.

    training: boolean
        if True, all models will be loaded right from the constructor, so that
        they will be available for back propagation and training.
        If false, the models will only be loaded when required, saving RAM
        usage.

    device: {torch device | 'cpu'|'cuda'}
        The device on which to create the separator

    smart_input_management: boolean
        whether or not to try smart management of the shapes and type of the
        audio input. This includes:
        -  conversion to pytorch
        -  if input is 1D, adding the samples and channels dimensions.
        -  if input is 2D
            o and the smallest dimension is 1 or 2, adding the samples one.
            o and all dimensions are > 2, assuming the smallest is the samples
              one, and adding the channel one
        - at he end, if the number of channels is greater than the number
          of time steps, swap those two.

        if the samples dimension is added, then it is removed from the output.
    """
```



> __Caution__ The `training` mode is incompatible with using the EM algorithm (`niter>0`). Only plain post-processing is supported right now for gradient computation. This is because the speed overhead of avoiding all the in-places operations was too large.
