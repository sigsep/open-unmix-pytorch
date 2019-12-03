import os
import pytest
import musdb
import simplejson as json
import museval
import numpy as np
import eval

test_track = 'Al James - Schoolboy Facination'

json_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/%s.json' % test_track,
)


@pytest.fixture()
def mus():
    return musdb.DB(download=True)


def test_estimate_and_evaluate(mus):
    # return any number of targets
    with open(json_path) as json_file:
        ref = json.loads(json_file.read())

    track = [track for track in mus.tracks if track.name == test_track][0]

    scores = eval.separate_and_evaluate(
        track,
        targets=['vocals', 'drums', 'bass', 'other'],
        model_name='umx',
        niter=1,
        alpha=1,
        softmask=False,
        output_dir=None,
        eval_dir=None,
        device='cpu'
    )

    assert scores.validate() is None

    with open(
        os.path.join('.', track.name) + '.json', 'w+'
    ) as f:
        f.write(scores.json)

    scores = json.loads(scores.json)

    for target in ref['targets']:
        for metric in ['SDR', 'SIR', 'SAR', 'ISR']:

            ref = np.array([d['metrics'][metric] for d in target['frames']])
            idx = [t['name'] for t in scores['targets']].index(target['name'])
            est = np.array(
                [
                    d['metrics'][metric]
                    for d in scores['targets'][idx]['frames']
                ]
            )

            assert np.allclose(ref, est, atol=1e-02)
