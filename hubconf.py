# This file is to be parsed by torch.hub mechanics
#
# `xxx_spec` take spectrogram inputs and output separated spectrograms
# `xxx`      take waveform inputs and output separated waveforms

# Optional list of dependencies required by the package
dependencies = ['torch', 'numpy']

from umx import umxse_spec
from umx import umxse

from umx import umxhq_spec
from umx import umxhq

from umx import umx_spec
from umx import umx
