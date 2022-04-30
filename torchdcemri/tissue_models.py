import torch
import torch.nn.functional as F

from torch import Tensor


def patlak(t, AIF, Kt, Vp, t_onset=0) -> Tensor:
    """Computes concentration over time, from batch of Kt and Vp, using the Patlak model.
    :param t: Timestamps to output concentration values for (s)
    :param AIF: AIF instance with .sample method
    :param Kt: Transfer constant (1/min)
    :param Vp: Fractional volume of blood plasma in tissue
    :param t_onset: Time of bolus arrival (s)
    :return: Concentration of contrast agent over time
    """
    t -= t_onset
    Cp = torch.from_numpy(AIF.sample(t)).to(Kt.device)
    Ct = torch.cumulative_trapezoid(F.pad(Cp, (1, 0), value=0), F.pad(t, (1, 0), value=0))
    return torch.outer(Cp, Vp) + torch.outer(Ct, Kt)


def tofts(t, AIF, Kt, Ve, Vp, t_onset=0, Kep=None, extended=False) -> Tensor:
    """Computes concentration over time, from batch of Kt, Ve, Vp, using the (Extended) Tofts-Kety model.
    :param t: Timestamps to output concentration values for (s)
    :param AIF: AIF instance with .sample method
    :param Kt: Transfer constant (1/min)
    :param Ve: Fractional volume of extravascullar extracellular space
    :param Vp: Fractional volume of blood plasma in tissue
    :param t_onset: Time of bolus arrival (s)
    :return: Concentration of contrast agent over time
    """
    t -= t_onset
    Ct = torch.zeros((len(t), len(Kt)), device=Kt.device)
    Cp = torch.from_numpy(AIF.sample(t)).to(Kt.device)

    if Kep is None:
        Kep = torch.zeros_like(Kt, device=Kt.device)
        Kep[Ve != 0] = Kt[Ve != 0] / Ve[Ve != 0]

    for k in range(len(t)):
        if t[k] <= 0:
            continue
        integ = Cp[:k+1] * torch.exp(-torch.outer(Kep, t[k] - t[:k+1]))
        Ct[k] = torch.trapz(integ, t[:k+1])
    if extended:
        return torch.outer(Cp, Vp) + Kt * Ct
    return Kt * Ct


def extended_tofts(t, AIF, Kt, Ve, Vp, t_onset=0, Kep=None) -> Tensor:
    """Alias for tofts(args, extended=True)."""
    return tofts(t, AIF, Kt, Ve, Vp, t_onset=t_onset, Kep=Kep, extended=True)
