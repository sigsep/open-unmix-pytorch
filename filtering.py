import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import warnings
import model
import utils
import torchaudio
from torchaudio.functional import istft

# Define basic complex operations on torch.Tensor objects whose last dimension
# consists in the concatenation of the real and imaginary parts.

def _norm(x):
    r"""Computes the norm value of a torch Tensor, assuming that it
    comes as real and imaginary part in its last dimension.

    Parameters:
    ----------
    x: torch Tensor [shape=(..., 2)]

    Returns:
    torch Tensor, with shape as x excluding the last dimension.
    """
    return torch.abs(x[..., 0])**2 + torch.abs(x[..., 1])**2

def _mul_add(a, b, out):
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""

    # check `out` and allocate it if needed
    target_shape = torch.Size([
        max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (
            real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (
            real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out

def _mul(a, b, out=None):
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts
    can work in place in case out is a only"""
    target_shape = torch.Size([
        max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return out


def _inv(z, out=None):
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts.
    can work in place in case out is z"""
    ez = _norm(z)
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez
    return out


def _conj(z, out=None):
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts.
    can work in place in case out is z"""
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]
    return out


def _invert(M, out=None):
    """
    Invert 1x1 or 2x2 matrices

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Parameters
    ----------
    M: torch.Tensor [shape=(..., nb_channels, nb_channels, 2)]
        matrices to invert: must be square along dimensions -3 and -2

    Returns
    -------
    invM: torch.Tensor, [shape=M.shape]
        inverses of M
    """
    nb_channels = M.shape[-2]

    if out is None or out.shape != M.shape:
        out = torch.empty_like(M)

    if nb_channels == 1:
        # scalar case
        out = _inv(M, out)
    elif nb_channels == 2:
        # two channels case: analytical expression

        # first compute the determinent
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        # invert it
        invDet = _inv(det)

        # then fill out the matrix with the inverse
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else:
        raise Exception('Only 2 channels are supported for the torch version.')
    return out


# Now define the signal-processing low-level functions used by the Separator

def expectation_maximization(y, x, iterations=2, verbose=0, eps=None):
    r"""Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Parameters
    ----------
    y: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
        initial estimates for the sources

    x: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
        complex STFT of the mixture signal

    iterations: int [scalar]
        number of iterations for the EM algorithm.

    verbose: boolean
        display some information if True

    eps: float or None [scalar]
        The epsilon value to use for regularization and filters.
        If None,  the default will use the epsilon of torch.real(x) dtype.

    Returns
    -------
    y: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
        estimated sources after iterations

    v: torch.Tensor [shape=(nb_frames, nb_bins, nb_sources)]
        estimated power spectral densities

    R: torch.Tensor [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
        estimated spatial covariance matrices


    Note
    -----
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.

        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning
    -------
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.

    """
    # to avoid dividing by zero
    if eps is None:
        eps = torch.tensor(torch.finfo(x.dtype).eps,
                           dtype=x.dtype, device=x.device)

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape[:-1]
    nb_sources = y.shape[-1]

    if verbose:
        print('Number of iterations: ', iterations)

    regularization = torch.cat(
        (
            torch.eye(2, dtype=x.dtype, device=x.device)[..., None],
            torch.zeros((2, 2, 1), dtype=x.dtype, device=x.device)
        ),
        dim=2
    )
    regularization = torch.sqrt(eps)*(
        regularization[None, None, ...].expand(
                            (-1, nb_bins, -1, -1, -1))
    )

    # allocate the spatial covariance matrices
    R = [
        torch.zeros(
            (nb_bins, nb_channels, nb_channels, 2),
            dtype=x.dtype, device=x.device
        )
        for j in range(nb_sources)
    ]
    weight = torch.zeros(
        (nb_bins,),
        dtype=x.dtype, device=x.device
    )

    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        if verbose:
            print('EM, iteration', (it+1))

        # update the spectrograms
        v = torch.mean(
            torch.abs(y[..., 0, :])**2 + torch.abs(y[..., 1, :])**2,
            dim=-2
        )

        for j in range(nb_sources):
            R[j][...] = 0
            weight[...] = eps
            for t in torch.utils.data.DataLoader(torch.arange(nb_frames),
                                                batch_size=200):
                R[j] = R[j] + torch.sum(
                    _covariance(y[t, ..., j]),
                    dim=0
                )
                weight = weight + torch.sum(v[t, ..., j], dim=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)

        # cloning y if we track gradient, because we're going to update it
        if y.requires_grad:
            y = y.clone()

        for t in torch.utils.data.DataLoader(torch.arange(nb_frames),
                                             batch_size=200):
            y[t,...] = 0
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + (
                    v[t, ..., j, None, None, None]
                    * R[j][None, ...].clone()
                )

            inv_Cxx = _invert(Cxx)
            
            # separate the sources
            for j in range(nb_sources):
                gain = torch.zeros_like(inv_Cxx)
                # computes multichannel Wiener gain as v_j R_j inv_Cxx
                for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
                    gain[..., i1, i2, :] = _mul_add(
                            R[j][None, :, i1, i3, :].clone(),
                            inv_Cxx[..., i3, i2, :],
                            gain[..., i1, i2, :]
                    )
                gain = gain * v[t, ..., None, None, None, j]

                # apply it to the mixture
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(
                        gain[..., i, :],
                        x[t, ..., i, None, :],
                        y[t, ..., j]
                    )

    return y, v, R


def wiener(v, x, iterations=1, use_softmask=True, build_residual=False, eps=None):
    """Wiener-based separation for multichannel audio.

    The method uses the (possibly multichannel) spectrograms `v` of the
    sources to separate the (complex) Short Term Fourier Transform `x` of the
    mix. Separation is done in a sequential way by:

    * Getting an initial estimate. This can be done in two ways: either by
      directly using the spectrograms with the mixture phase, or
      by using :func:`softmask`.

    * Refinining these initial estimates through a call to
      :func:`expectation_maximization`.

    This implementation also allows to specify the epsilon value used for
    regularization. It is based on [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [4] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Parameters
    ----------

    v: torch.Tensor [shape=(nb_frames, nb_bins, {1,nb_channels}, nb_sources)]
        spectrograms of the sources. This is a nonnegative tensor that is
        usually the output of the actual separation method of the user. The
        spectrograms may be mono, but they need to be 4-dimensional in all
        cases.

    x: torch.Tensor [complex, shape=(nb_frames, nb_bins, nb_channels, 2)]
        STFT of the mixture signal.

    iterations: int [scalar]
        number of iterations for the EM algorithm

    use_softmask: boolean
        * if `False`, then the mixture phase will directly be used with the
          spectrogram as initial estimates.

        * if `True`, a softmasking strategy will be used as described in
          :func:`softmask`.

    build_residual: boolean
        if `True`, an additional target is created, which is
        equal to the mixture minus the estimated targets, before application of
        expectation maximization
    eps: {None, float}
        Epsilon value to use for computing the separations. This is used
        whenever division with a model energy is performed, i.e. when
        softmasking and when iterating the EM.
        It can be understood as the energy of the additional white noise
        that is taken out when separating.
        If `None`, the default value is taken as `torch.finfo(x.dtype).eps`.

    Returns
    -------

    y: torch.Tensor
            [complex, shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
        STFT of estimated sources

    Note
    ----

    * Be careful that you need *magnitude spectrogram estimates* for the
      case `softmask==False`.
    * We recommand to use `softmask=False` only if your spectrogram model is
      pretty good, e.g. when the output of a deep neural net. In the case
      it is not so great, opt for an initial softmasking strategy.
    * The epsilon value will have a huge impact on performance. If it's large,
      only the parts of the signal with a significant energy will be kept in
      the sources. This epsilon then directly controls the energy of the
      reconstruction error.

    Warning
    -------
    As in :func:`expectation_maximization`, we recommend converting the
    mixture `x` to double precision `torch.float64` *before* calling
    :func:`wiener`.

    """
    if use_softmask:
        y = softmask(v, x, eps=eps)
    else:
        angle = torch.atan2(x[..., 1], x[..., 0])[..., None]
        nb_sources = v.shape[-1]
        y = torch.zeros(x.shape + (nb_sources,), dtype=x.dtype,
                        device=x.device)
        y[..., 0, :] = v * torch.cos(angle)
        y[..., 1, :] = v * torch.sin(angle)

    if build_residual:
        # if required, adding an additional target as the mix minus
        # available targets
        y = torch.cat(
            [y, x[..., None] - y.sum(dim=-1, keepdim=True)],
            dim=-1
        )

    if not iterations:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = torch.max(torch.tensor(1., dtype=x.dtype, device=x.device),
                        torch.sqrt(_norm(x)).max()/10.)
    x = x / max_abs
    y = y / max_abs

    # call expectation maximization
    y = expectation_maximization(y, x, iterations, eps=eps)[0]

    # scale estimates up again
    y = y * max_abs
    return y


def softmask(v, x, eps=None):
    """Separates a mixture with a ratio mask, using the provided sources
    spectrograms estimates. Additionally allows compressing the mask with
    a logit function for soft binarization.
    The filter does *not* take multichannel correlations into account.

    The masking strategy can be traced back to the work of N. Wiener in the
    case of *power* spectrograms [1]_. In the case of *fractional* spectrograms
    like magnitude, this filter is often referred to a "ratio mask", and
    has been shown to be the optimal separation procedure under alpha-stable
    assumptions [2]_.

    References
    ----------
    .. [1] N. Wiener,"Extrapolation, Inerpolation, and Smoothing of Stationary
        Time Series." 1949.

    .. [2] A. Liutkus and R. Badeau. "Generalized Wiener filtering with
        fractional power spectrograms." 2015 IEEE International Conference on
        Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

    Parameters
    ----------
    v: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
        spectrograms of the sources

    x: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
        mixture signal

    Returns
    -------
    torch.Tensor, shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)
        estimated sources

    """
    # to avoid dividing by zero
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    # create the soft mask as the ratio of the spectrograms with their sum
    return x[..., None] * (
        v / (eps + torch.sum(v, dim=-1, keepdim=True).to(x.dtype))
    )[..., None, :]



def _covariance(y_j):
    """
    Compute the empirical covariance for a source.

    Parameters
    ----------
    y_j: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)].
          complex stft of the source.

    Returns
    -------
    Cj: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
        just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]
    Cj = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels, 2),
                     dtype=y_j.dtype,
                     device=y_j.device)
    for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
        Cj[..., i1, i2, :] = _mul_add(
            y_j[..., i1, :],
            _conj(y_j[..., i2, :]),
            Cj[..., i1, i2, :]
        )
    return Cj


# now we are ready to define the actual Separator Module


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Parameters
    ----------
    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage. Zeroed if only one target is estimated. 
         defaults to 1. 

    softmask: boolean
        if activated and strictly more than one models are available,
        then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    build_residual: boolean
        adds an additional residual target, obtained by subtracting the estimated
        sources from the mixture, before any potential EM post-processing.
        If only one model is present, this additional target is called `accompaniment`.
        Otherwise, it is called `residual`.
        Defaults to False

    batch_size: {None | int}
        The size of the batches (number of frames) on which to apply filtering
        independently. This means assuming time varying stereo models and
        localization of sources.
        None means not batching but using the whole signal. It comes at the
        price of a much larger memory usage.

    preload: boolean
        if True, all models will be loaded right from the constructor, so that
        they will be available for back propagation and training.
        If false, the models will only be loaded on the fly when required, saving RAM
        usage, but preventing backpropagation.

    device: {torch device | 'cpu'|'cuda'}
        The device on which to create the separator
    """
    def __init__(self, targets, model_name='umxhq',
                 niter=1, softmask=False, alpha=1.0, build_residual=False,
                 batch_size=None, preload=False, device='cpu'):
        super(Separator, self).__init__()
        if not utils._torchaudio_available():
            raise Exception('The Separator class only works when torchaudio '
                            'is available.')

        # saving parameters
        self.device = device
        self.targets = targets
        self.model_name = model_name
        self.preload = preload
        self.niter = niter
        self.alpha = alpha
        self.softmask = softmask
        self.build_residual = build_residual
        self.batch_size = batch_size

        # loading all models in the case of training
        if self.preload:
            self.target_models = nn.ModuleList([
                model.load_model(
                    target=target,
                    model_name=self.model_name,
                    device=self.device
                ) for target in self.targets
            ])
        else:
            self.target_models = None

    def freeze(self):
        for target_model in self.target_models:
            target_model.freeze()

    def forward(self, audio, rate):
        """
        Performing the separation on audio input

        Parameters
        ----------
        audio: torch.Tensor [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

        rate: int
        sampling rate of the audio.

        Returns
        -------
        estimates: `dict` [`str`, `torch.Tensor`
                                  shape(nb_samples, nb_channels, nb_timesteps)]
            dictionary of all restimates as performed by the separation model.
            Note that there may be an additional source in case
            a residual source is added.
        model_rate: int
            the new sampling rate, desired by the open-unmix model
            """
        if audio.requires_grad and not self.preload:
            raise Exception(
                "For computing gradients, the Separator must be used with pre-loading"
            )

        # initializing spectrograms variable
        V = None

        nb_sources = len(self.targets)
        nb_samples = audio.shape[0]

        for j, target in enumerate(self.targets):
            if not self.preload:
                # if the models have not been pre-loaded, load it now
                unmix_target = model.load_model(
                    target=target,
                    model_name=self.model_name,
                    device=self.device
                )
                # we don't record gradient in that case
                unmix_target.freeze()
            else:
                unmix_target = self.target_models[j]

            # handle possibly different sample rates
            model_rate = unmix_target.sample_rate.item()
            if rate != model_rate:
                # we have to resample to model samplerate if needed
                # this makes sure we resample input only once
                model_rate = unmix_target.sample_rate.item()
                resampler = torchaudio.transforms.Resample(
                    orig_freq=rate,
                    new_freq=model_rate).to(self.device)
                audio = resampler(audio)
                """audio = torch.cat(
                    [resampler(audio[sample])[None, ...]
                     for sample in range(nb_samples)],
                    dim=0)"""
                rate = model_rate

            # apply current model to get the source spectrogram
            Vj = unmix_target(audio)

            if self.softmask:
                # only exponentiate the model if we use softmask
                Vj = Vj**self.alpha

            # output is nb_frames, nb_samples, nb_channels, nb_bins
            if V is None:
                V = torch.zeros(Vj.shape + (nb_sources,), dtype=torch.float64,
                                device=Vj.device)
            V[..., j] = Vj

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        V = V.permute(1, 0, 3, 2, 4).to(torch.float64)

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        X = unmix_target.stft(audio).to(torch.float64)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        X = X.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build the accompaniment
        targets = self.targets
        if self.build_residual:
            targets += (['residual'] if nb_sources > 1
                        else ['accompaniment'])
            nb_sources += 1

        nb_frames = V.shape[1]
        Y = torch.zeros(X.shape + (nb_sources, ), dtype=torch.float32,
                        device=X.device)
        frames_loader = ([torch.arange(nb_frames), ] if self.batch_size is None
                         else DataLoader(torch.arange(nb_frames),
                                         batch_size=self.batch_size))
        for sample in range(nb_samples):
            for t in frames_loader:            
                Y[sample, t] = wiener(V[sample, t], X[sample, t],
                                      self.niter, use_softmask=self.softmask,
                                      build_residual=self.build_residual
                                      ).to(torch.float32)
        estimates = {}

        # getting to (nb_samples, channel, fft_size, n_frames, 2, nb_sources)
        Y = Y.permute(0, 3, 2, 1, 4, 5)

        # Now performing the inverse STFTs
        for j, name in enumerate(targets):
            estimates[name] = torch.cat([
                istft(
                    Y[sample, ..., j],
                    n_fft=unmix_target.stft.n_fft,
                    hop_length=unmix_target.stft.n_hop,
                    window=unmix_target.stft.window,
                    center=unmix_target.stft.center,
                    normalized=False, onesided=True,
                    pad_mode='reflect', length=audio.shape[-1]
                ).transpose(0, 1)[None, ...]
                for sample in range(nb_samples)], dim=0)

        return estimates, model_rate
