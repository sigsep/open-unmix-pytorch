from openunmix.utils import AverageMeter, EarlyStopping


def test_average_meter():
    losses = AverageMeter()
    losses.update(1.0)
    losses.update(3.0)
    assert losses.avg == 2.0


def test_early_stopping():
    es = EarlyStopping(patience=2)
    es.step(1.0)

    assert not es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)
