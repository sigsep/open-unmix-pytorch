

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--targets list(str)`           | Targets to be used for separation. For each target a model file with with same name is required.                                                  | `['vocals', 'drums', 'bass', 'other']`          |
| `--residual`           |               computes a residual target, for custom separation scenarios when not all targets are available (at the expense of slightly less performance). E.g vocal/accompaniment can be performed with `--targets vocals --residual`.                                   | not set          |
| `--softmask`       | if activated, then the initial estimates for the sources will be obtained through a ratio mask of the mixture STFT, and not by using the default behavior of reconstructing waveforms by using the mixture phase.  | not set            |
| `--niter <int>`           | Number of EM steps for refining initial estimates in a post-processing stage. `--niter 0` skips this step altogether. More iterations can get better interference reduction at the price of more artifacts.                                                  | `1`          |
| `--alpha <float>`         |In case of softmasking, this value changes the exponent to use for building ratio masks. A smaller value usually leads to more interference but better perceptual quality, whereas a larger value leads to less interference but an "overprocessed" sensation.                                                          | `1.0`            |
