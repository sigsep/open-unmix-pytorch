import numpy as np
import pescador
import random


def excerpt_sampler(
    dataset, idx,
    ex_length=256, ex_hop=256,
    shuffle=True,
    seed=42
):
    x, y = dataset[idx]
    nb_timesteps = x.shape[0]

    # get all t indices and shuffle, make sure that excerpt is shorter than
    # number of frames
    if ex_length < nb_timesteps:
        steps = np.arange(0, nb_timesteps - ex_length, ex_hop)
    else:
        steps = [0]
        ex_length = nb_timesteps

    if shuffle:
        # shuffle t indices in place
        np.random.seed(seed)
        np.random.shuffle(steps)

    for s in steps:
        cur_x = x[s:s+ex_length, ...]
        cur_y = y[s:s+ex_length, ...]
        yield dict(x=cur_x, y=cur_y)


def mixmux(
    dataset,
    batch_size,
    ex_length=256,
    ex_hop=256,
    shuffle=True,
    seed=42,
    active_streamers=100,
):
    streams = []
    for idx in range(len(dataset)):
        s = pescador.Streamer(
            excerpt_sampler,
            dataset, idx,
            ex_length=ex_length,
            ex_hop=ex_hop,
            shuffle=shuffle,
            seed=seed,
        )
        streams.append(s)

    if not shuffle:
        mux = pescador.ChainMux(streams, mode='exhaustive')
    else:
        random.seed(seed)
        # shuffle the streams, just in case
        random.shuffle(streams)

        mux = pescador.StochasticMux(
            streams,
            active_streamers,
            rate=None,
            mode='exhaustive',
            random_state=seed
        )

    # get `batch_size` samples from the mux
    batches = pescador.Streamer(pescador.buffer_stream, mux, batch_size)

    return batches
