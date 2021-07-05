# This file is to be parsed by torch.hub mechanics
#
# `xxx_spec` take spectrogram inputs and output separated spectrograms
# `xxx`      take waveform inputs and output separated waveforms

# Optional list of dependencies required by the package
dependencies = ['torch', 'numpy']

from openunmix import umxse_spec
from openunmix import umxse

from openunmix import umxhq_spec
from openunmix import umxhq

from openunmix import umx_spec
from openunmix import umx

from openunmix import umxl_spec
from openunmix import umxl
