import numpy as np

from abc import ABC, abstractmethod
from typing import Optional


class AIFBase(ABC):
    """Abstract base class for AIF implementations."""
    @abstractmethod
    def fit(self, AIF: np.array, t: np.array):
        pass

    @abstractmethod
    def sample(self, t: np.array) -> np.array:
        pass


class InterpolatedAIF(AIFBase):
    """Arterial Input Function, uses linear interpolation for sampling."""
    def __init__(self, AIF: Optional[np.array] = None, t: Optional[np.array] = None):
        """Defaults to population based AIF when not provided with an AIF.
        :param AIF: array_like containing AIF sample values
        :param t: array_like with timestamps corresponding to AIF
        """
        if AIF is not None:
            assert t is not None
            self.fit(AIF, t)
        else:
            self.AIF, self.t = _default_population_based_AIF()

    def fit(self, AIF: np.array, t: np.array):
        self.AIF = AIF
        self.t = t

    def sample(self, t: np.array) -> np.array:
        """Sample from the AIF using interpolation.
        :param t: array_like with timesteps to sample, must be strictly increasing
        :return: AIF values at given timestamps
        """
        return np.interp(t, self.t, self.AIF)


def _default_population_based_AIF() -> np.array:
    """Returns population based AIF and timesteps as tuple."""
    AIF_pop = np.array([
        0.0, 0.0908, 0.1787, 0.5667, 1.9445, 4.6015, 6.5225, 8.7509, 6.9091, 6.0901, 4.6583, 3.5056,
        2.8350, 2.3160, 1.9862, 1.8966, 1.8934, 2.0551, 2.1454, 2.1860, 2.1717, 2.1783, 2.0682, 1.9374, 1.8774,
        1.8100, 1.7271, 1.6808, 1.6205, 1.6182, 1.5581, 1.5831, 1.5469, 1.5513, 1.5469, 1.5393, 1.5398, 1.5243,
        1.5588, 1.5137, 1.4882, 1.5302, 1.4465, 1.4588, 1.4582, 1.4741, 1.4251, 1.4850, 1.4249, 1.4075, 1.4414,
        1.4190, 1.3938, 1.4047, 1.3961, 1.3857, 1.3794, 1.3691, 1.3585, 1.3784, 1.3299, 1.3100, 1.3231, 1.3130,
        1.3053, 1.3129, 1.2640, 1.3032, 1.2721, 1.2649, 1.2426, 1.2345, 1.2416, 1.2340, 1.2172, 1.2128, 1.2374,
        1.2213, 1.2012, 1.1947, 1.1851, 1.1543, 1.1841, 1.1892, 1.1636, 1.1522, 1.2028, 1.1482, 1.1751, 1.1605,
        1.1531, 1.1628, 1.1232, 1.1255, 1.1227, 1.1362, 1.1002, 1.1486, 1.1178, 1.0933, 1.1085, 1.0972, 1.0982,
        1.0860, 1.1026, 1.0679, 1.0670, 1.0870, 1.0640, 1.0839, 1.0671, 1.0452, 1.0639, 1.0490, 1.0358, 1.0360,
        1.0378, 1.0258, 1.0371, 1.0287, 1.0300, 1.0171, 1.0195, 1.0044, 1.0052, 1.0154, 1.0510, 1.0066, 1.0161,
        0.9986, 0.9653, 0.9922, 0.9914, 0.9749, 0.9934, 0.9621, 0.9733, 0.9894, 0.9900, 0.9754, 0.9240, 0.9347,
        0.8556, 0.7921, 0.7380, 0.6348
    ])
    t = np.arange(len(AIF_pop)) * 1.75  # t in seconds, belonging to the AIF samples
    return AIF_pop, t
