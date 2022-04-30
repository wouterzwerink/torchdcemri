import torch
import numpy as np
import math

from typing import Union, Optional
from torch import Tensor


def S0_to_M0(S0: Union[float, Tensor], TR: float, T10: Union[float, Tensor], FA: float) -> Union[float, Tensor]:
    """Compute fully relaxed signal from mean signal before bolus arrival.
    :param S0: Mean signal before bolus arrival
    :param TR: Repetition time (s)
    :param T10: Baseline T1
    :param FA: Flip angle (rad)
    :return: Fully relaxed signal for 90 degree pulse when TR >> T10
    """
    exp = torch.exp(-TR * 1 / T10) if torch.is_tensor(T10) else math.exp(-TR / T10)
    return S0 * (1 - exp * np.cos(FA)) / (np.sin(FA) * (1 - exp))


def concentration_to_R1(Ct: Tensor, T10: Union[Tensor, float], r1: Union[Tensor, float]) -> Tensor:
    """Converts concentration over time to T1 over time.
    :param Ct: Concentration of contrast agent over time
    :param r1: Relaxation rate of the tissue
    :param T10: Baseline T1 value / map
    :return: R1 over time
    """
    return Ct * r1 + 1 / T10


def R1_to_DCE_signal(R1: Tensor, TR: float, FA: float,
                     T10: Union[Tensor, float], S0: Optional[Union[Tensor, float]] = None) -> Tensor:
    """Convert R1 map to DCE signal using SPGR equation."""
    M0 = 1 if S0 is None else S0_to_M0(S0, TR, T10, FA)
    return SPGR(R1, TR, FA, M0)


def SPGR(R1: Tensor, TR: float, FA: float, M0: Optional[Union[Tensor, float]] = 1) -> Tensor:
    """Spoiled gradient echo signal equation.
    :param R1: Relaxation rate (1 / T1)
    :param TR: Repitition time (s)
    :param FA: Flip angle (rad)
    :param M0: Fully relaxed signal for 90 degree pulse when TR >> T10
    :return: DCE-signal
    """
    exp = torch.exp(-R1 * TR)
    return M0 * (((1 - exp) * np.sin(FA)) / (1 - np.cos(FA) * exp))


def radial_trajectories(spokelength: int, nspokes: int, angle: float, device: Optional[torch.device] = None) -> Tensor:
    """Compute radial sampling trajectories.
    :param spokelength: Length of a single spoke
    :param nspokes: Number of spokes
    :param angle: Angle between consecutive spokes (rad)
    :return: Trajectories (2, samples), scaled between -pi and pi
    """
    angles_radian = np.mod(np.arange(nspokes) * angle, 2 * np.pi)
    rho = np.linspace(-spokelength / 2, spokelength / 2, spokelength)
    X = np.outer(-rho, np.sin(angles_radian))
    Y = np.outer(rho, np.cos(angles_radian))
    highest = max(np.max(np.abs(X)), np.max(np.abs(Y)))
    X = np.pi * X / highest
    Y = np.pi * Y / highest
    return torch.tensor(np.stack((X.T.flatten(), Y.T.flatten()), axis=0), device=device)


def golden_angle_radial_trajectories(spokelength: int, nspokes: int, device: Optional[torch.device] = None) -> Tensor:
    """Computes trajectory for golden angle radial sampling."""
    golden_angle = np.pi / ((1 + 5 ** 0.5) / 2)  # Pi / golden ratio
    return radial_trajectories(spokelength, nspokes, golden_angle, device=device)


if __name__ == "__main__":
    pass
