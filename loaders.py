import numpy as np
import random
import pescador


def excerpt_sampler(
    dataset, idx,
    ex_length=256, ex_hop=256,
    shuffle=True,
    seed=42
):
    x, y = dataset[idx]
    nb_channels, nb_timesteps = x.shape
    # get all t indices and shuffle, make sure that excerpt is shorter than
    # number of samples
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
        cur_x = x[..., s:s+ex_length]
        cur_y = y[..., s:s+ex_length]
        yield dict(x=cur_x, y=cur_y)


class PescadorSampler(object):
    def __init__(
        self,
        dataset,
        batch_size=16,
        ex_length=256,
        ex_hop=256,
        shuffle=True,
        seed=42,
        active_streamers=10
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ex_length = ex_length
        self.ex_hop = ex_hop
        self.shuffle = shuffle
        self.seed = seed
        self.active_streamers = active_streamers
        self.streams = self.fill_streams()

        if not shuffle:
            self.mux = pescador.ChainMux(self.streams, mode='exhaustive')
        else:
            random.seed(seed)
            # shuffle the streams, just in case
            random.shuffle(self.streams)

            self.mux = pescador.StochasticMux(
                self.streams,
                self.active_streamers,
                rate=None,
                mode='exhaustive',
                random_state=seed
            )

        # get `batch_size` samples from the mux
        self.batches = pescador.Streamer(pescador.buffer_stream, self.mux, self.batch_size)

    def fill_streams(self):
        streams = []
        for idx in range(len(self.dataset)):
            s = pescador.Streamer(
                excerpt_sampler,
                self.dataset, idx,
                ex_length=self.ex_length,
                ex_hop=self.ex_hop,
                shuffle=self.shuffle,
                seed=self.seed
            )
            streams.append(s)
        return streams

    def __iter__(self):
        return self.batches.iterate()


if __name__ == "__main__":
    import data
    # dataset iterator test
    dataset = data.MUSDBDataset(
        download=True, subsets="train", validation_split='train'
    )
    print(len(dataset))
    train = PescadorSampler(dataset)
    for x, y in train:
        print(x.shape)