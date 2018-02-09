from mxnet.initializer import Xavier
import numpy as np


class IdentityBias(Xavier):
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(IdentityBias, self).__init__(rnd_type=rnd_type, factor_type=factor_type,
                                           magnitude=magnitude)

    def _init_bias(self, _, arr):
        size = arr.size
        width = int(np.sqrt(size))
        assert width ** 2 == size
        arr[:] = np.eye(width).flatten()
