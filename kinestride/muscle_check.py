"""Small analytic oracle for the Kinestride muscle equations."""
from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class Params:
    fmax: float = 100.0
    lopt: float = 1.0
    width: float = 0.25
    vmax: float = 1.0
    eccentric_cap: float = 1.5
    kpassive: float = 10.0
    lslack: float = 1.1
    tau_rise: float = 0.04
    tau_fall: float = 0.07
    tau_fatigue: float = 8.0
    tau_recovery: float = 20.0


def activation(a, u, dt, p=Params()):
    tau = p.tau_rise if u > a else p.tau_fall
    return u + (a - u) * exp(-dt / tau)


def fatigue(f, output, dt, p=Params()):
    target = min(1.0, max(0.0, output))
    tau = p.tau_fatigue if target > 0 else p.tau_recovery
    return target + (f - target) * exp(-dt / tau)


def length_factor(length, p=Params()):
    x = (length - p.lopt) / p.width
    return exp(-(x * x))


def velocity_factor(speed, p=Params()):
    x = speed / p.vmax
    if x < 0:
        return max(0.0, 1.0 + x / (1.0 - x))
    return min(p.eccentric_cap, 1.0 + (p.eccentric_cap - 1.0) * x / (1.0 + x))


def force(a, f, length, speed, p=Params()):
    active = a * (1.0 - f) * p.fmax * length_factor(length, p) * velocity_factor(speed, p)
    return active + p.kpassive * max(0.0, length - p.lslack) ** 2


def check():
    p = Params()
    assert abs(activation(0, 1, p.tau_rise) - (1 - exp(-1))) < 1e-12
    assert abs(activation(1, 0, p.tau_fall) - exp(-1)) < 1e-12
    assert abs(fatigue(0, 1, p.tau_fatigue) - (1 - exp(-1))) < 1e-12
    assert abs(fatigue(1, 0, p.tau_recovery) - exp(-1)) < 1e-12
    assert abs(force(1, 0, p.lopt, 0, p) - p.fmax) < 1e-12
    assert velocity_factor(-0.5, p) < 1 < velocity_factor(0.5, p) < p.eccentric_cap
    one = activation(0.2, 0.9, 0.12, p)
    two = activation(activation(0.2, 0.9, 0.03, p), 0.9, 0.09, p)
    assert abs(one - two) < 1e-12
    print("PASS muscle oracle: activation rise/fall, fatigue/recovery, isometric, FV, dt semigroup")


if __name__ == "__main__":
    check()
