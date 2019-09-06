import torch
import itertools


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


def _mul(a, b):
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts"""
    return torch.cat(
        ((a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])[..., None],
         (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])[..., None]),
        dim=-1)


def _inv(z):
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts."""
    ez = _norm(z)
    return torch.cat(
        ((z[..., 0] / ez)[..., None],
         (-z[..., 1] / ez)[..., None]),
        dim=-1)


def _conj(z):
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts."""
    return torch.cat((z[..., 0, None], -z[..., 1, None]), dim=-1)


def residual_model(v, x, alpha=1, autoscale=False):
    r"""Compute a model for the residual based on spectral subtraction.

    The method consists in two steps:

    * The provided spectrograms are summed up to obtain the *input* model for
      the mixture. This *input* model is scaled frequency-wise to best
      fit with the actual observed mixture spectrogram.

    * The residual model is obtained through spectral subtraction of the
      input model from the mixture spectrogram, with flooring to 0.

    Parameters
    ----------
    v: torch.Tensor [shape=(nb_frames, nb_bins, {1, nb_channels}, nb_sources)]
        Estimated spectrograms for the sources

    x: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
        complex mixture

    alpha: float [scalar]
        exponent for the spectrograms `v`. For instance, if `alpha==1`,
        then `v` must be homogoneous to magnitudes, and if `alpha==2`, `v`
        must homogeneous to squared magnitudes.

    Returns
    -------
    v: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_sources+1)]
        Spectrograms of the sources, with an appended one for the residual.

    Note
    ----
    It is not mandatory to input multichannel spectrograms. However, the
    output spectrograms *will* be multichannel.

    Warning
    -------
    You must be careful to set `alpha` as the exponent that corresponds to `v`.
    In other words, *you must have*: ``torch.abs(x)**alpha`` homogeneous to
    `v`.
    """
    # to avoid dividing by zero
    eps = torch.tensor(torch.finfo(v.dtype).eps, device=v.device,
                       dtype=v.dtype)
    # spectrogram for the mixture
    vx = torch.max(eps, torch.sqrt(_norm(x))**alpha)

    # compute the total model as provided
    v_total = torch.sum(v, dim=-1)

    if autoscale:
        # quick trick to scale the provided spectrograms to fit the mixture
        nb_frames = x.shape[0]
        gain = 0
        weights = eps
        for t in range(nb_frames):
            gain = gain + vx[None, t] * v_total[None, t]
            weights = weights + v_total[None, t]**2
        gain = gain / weights
        v = v * gain[..., None]
        # compute the total model as provided
        v_total = torch.sum(v, dim=-1)

    # residual is difference between the observation and the model
    vr = torch.relu(vx - v_total)
    v = torch.cat((v, vr[..., None]), dim=3)
    return v


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

    # allocate the spatial covariance matrices and PSD
    R = torch.zeros((nb_bins, nb_channels, nb_channels, 2, nb_sources),
                    dtype=x.dtype, device=y.device)
    v = torch.zeros((nb_frames, nb_bins, nb_sources),
                    dtype=x.dtype, device=y.device)

    if verbose:
        print('Number of iterations: ', iterations)

    regularization = torch.cat((torch.eye(2)[..., None],
                                torch.zeros((2, 2, 1))),
                               dim=2).to(x.dtype).to(x.device)
    regularization = torch.sqrt(eps)*(
                        regularization[None, None, ...].expand(
                            (-1, nb_bins, -1, -1, -1)))
    for it in range(iterations):
        print(torch.max(_norm(y)))

        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        if verbose:
            print('EM, iteration %d' % (it+1))

        for j in range(nb_sources):
            # update the spectrogram model for source j
            v[..., j], R[..., j] = get_local_gaussian_model(
                y[..., j],
                eps)
            print(j, v[..., j].max())

        for t in range(nb_frames):
            Cxx = get_mix_model(v[None, t, ...], R)
            Cxx = Cxx + regularization
            inv_Cxx = _invert(Cxx)
            # separate the sources
            for j in range(nb_sources):
                W_j = wiener_gain(v[None, t, ..., j], R[..., j], inv_Cxx)
                y[t, ..., j] = apply_filter(x[None, t, ...], W_j)[0]
                print(t, 'before:', torch.norm(_norm(x[t, ...])).item(), 'after:',torch.norm(_norm(y[t, ...])).item())

    return y, v, R


def wiener(v, x, iterations=1, use_softmask=True, eps=None):
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
        angle = torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)
        y = v[..., None, :] * angle[..., None]
    if not iterations:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = torch.max(torch.tensor(1., dtype=x.dtype, device=x.device),
                        torch.sqrt(_norm(x)).max()/10.)
    import ipdb; ipdb.set_trace()
    y = expectation_maximization(y/max_abs, x/max_abs, iterations, eps=eps)[0]
    return y*max_abs


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
    total_energy = torch.sum(v, dim=-1, keepdim=True)
    filter = v/(eps + total_energy.to(x.dtype))
    return filter[..., None, :] * x[..., None]


def _invert(M):
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
    if nb_channels == 1:
        # scalar case
        invM = _inv(M)
    elif nb_channels == 2:
        # two channels case: analytical expression
        det = (
            _mul(M[..., 0, 0, :], M[..., 1, 1, :])
            - _mul(M[..., 0, 1, :], M[..., 1, 0, :]))
        invDet = _inv(det)
        invM = torch.empty_like(M)
        invM[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :])
        invM[..., 1, 0, :] = -_mul(invDet, M[..., 1, 0, :])
        invM[..., 0, 1, :] = -_mul(invDet, M[..., 0, 1, :])
        invM[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :])
    else:
        raise Exception('Only 2 channels are supported for the torch version.')
    return invM


def wiener_gain(v_j, R_j, inv_Cxx):
    """
    Compute the wiener gain for separating one source, given all parameters.
    It is the matrix applied to the mix to get the posterior mean of the source
    as in [1]_

    References
    ----------
    .. [1] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    Parameters
    ----------
    v_j: torch.Tensor [shape=(nb_frames, nb_bins)]
        power spectral density of the target source.

    R_j: torch.Tensor [shape=(nb_bins, nb_channels, nb_channels, 2)]
        spatial covariance matrix of the target source

    inv_Cxx: torch.Tensor
             [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
        inverse of the mixture covariance matrices

    Returns
    -------

    G: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
        wiener filtering matrices, to apply to the mix, e.g. through
        :func:`apply_filter` to get the target source estimate.

    """
    nb_channels = R_j.shape[1]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = torch.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2, :] = (G[..., i1, i2, :]
                             + _mul(R_j[None, :, i1, i3, :],
                                    inv_Cxx[..., i3, i2, :]))
    G = G * v_j[..., None, None, None]
    return G


def apply_filter(x, W):
    """
    Applies a filter on the mixture. Just corresponds to a matrix
    multiplication.

    Parameters
    ----------
    x: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
        STFT of the signal on which to apply the filter.

    W: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
        filtering matrices, as returned, e.g. by :func:`wiener_gain`

    Returns
    -------
    y_hat: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
        filtered signal
    """
    nb_channels = W.shape[-2]

    # apply the filter
    y_hat = torch.zeros_like(x)
    for i in range(nb_channels):
        y_hat = y_hat + _mul(W[..., i, :], x[..., i, None, :])
    return y_hat


    """    nb_channels = W.shape[-1]

        # apply the filter
        y_hat = 0+0j
        for i in range(nb_channels):
            y_hat += W[..., i] * x[..., i, None]
        return y_hat"""


def get_mix_model(v, R):
    """
    Compute the model covariance of a mixture based on local Gaussian models.
    simply adds up all the v[..., j] * R[..., j]

    Parameters
    ----------
    v: torch.Tensor [shape=(nb_frames, nb_bins, nb_sources)]
        Power spectral densities for the sources

    R: torch.Tensor [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
        Spatial covariance matrices of all sources

    Returns
    -------
    Cxx: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
        Covariance matrix for the mixture
    """
    nb_channels = R.shape[1]
    (nb_frames, nb_bins, nb_sources) = v.shape
    Cxx = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels, 2),
                      dtype=R.dtype, device=v.device)
    for j in range(nb_sources):
        Cxx = Cxx + v[..., j, None, None, None] * R[None, ..., j]
    return Cxx


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
        Cj[..., i1, i2] = (Cj[..., i1, i2]
                           + _mul(y_j[..., i1, :],
                                  _conj(y_j[..., i2, :])))

    return Cj


def get_local_gaussian_model(y_j, eps=1.):
    r"""
    Compute the local Gaussian model [1]_ for a source given the complex STFT.
    First get the power spectral densities, and then the spatial covariance
    matrix, as done in [1]_, [2]_

    References
    ----------
    .. [1] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [2] A. Liutkus and R. Badeau and G. Richard. "Low bitrate informed
        source separation of realistic mixtures." 2013 IEEE International
        Conference on Acoustics, Speech and Signal Processing. IEEE, 2013.

    Parameters
    ----------
    y_j: torch.Tensor [shape=(nb_frames, nb_bins, nb_channels, 2)]
          complex stft of the source.
    eps: float [scalar]
        regularization term

    Returns
    -------
    v_j: torch.Tensor [shape=(nb_frames, nb_bins)]
        power spectral density of the source
    R_J: torch.Tensor [shape=(nb_bins, nb_channels, nb_channels, 2)]
        Spatial covariance matrix of the source

    """
    v_j = torch.mean(_norm(y_j), dim=2)

    # updates the spatial covariance matrix
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]
    R_j = torch.zeros((nb_bins, nb_channels, nb_channels, 2),
                      dtype=y_j.dtype, device=y_j.device)
    weight = eps
    for t in range(nb_frames):
        R_j = R_j + _covariance(y_j[None, t, ...])
        weight = weight + v_j[None, t, ...]
    R_j = R_j / weight[..., None, None, None]
    return v_j, R_j
