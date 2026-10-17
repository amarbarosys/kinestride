# Kinestride muscles

Muscles add two float32 state values per muscle per body (8 bytes) while reusing fixed-tendon geometry. This note specifies a neural excitation model with exact first-order state updates, bounded force curves, and no MuJoCo-specific gain/bias/dyntype model.

## Activation and fatigue

For constant excitation `u` during `dt`, activation is

`a' = u + (a - u) exp(-dt / tau)`

Use `tau_rise` when `u > a`, otherwise `tau_fall`. This closed form has no explicit-Euler stability limit.

Fatigue `f` is in `[0,1]`. Let `p` be normalized active output at the start of the step. During excitation, `f_inf = p` and `tau = tau_fatigue`; when excitation is zero, `f_inf = 0` and `tau = tau_recovery`, including the activation tail as it falls:

`f' = f_inf + (f - f_inf) exp(-dt / tau)`

This exact constant-rate update separates depletion from slower recovery. It is operator split: force uses updated activation and starting fatigue; the following step sees updated fatigue.

## Force

Active force is `a (1-f) Fmax FL FV`. Length factor is Gaussian around optimum length: `FL = exp(-((l-lopt)/width)^2)`. Speed convention: `v > 0` means lengthening and `v < 0` means shortening. With normalized speed `x = v/vmax`, use `FV = max(0, (1+x)/(1-x/eccentric_cap))` for `-1 <= x <= 0`, and `FV = min(eccentric_cap, 1 + (eccentric_cap-1)x/(1+x))` for `x > 0`. The branches meet continuously at `FV(0)=1`; the shortening branch is exactly zero at `x=-1` and rises toward one as shortening speed approaches zero. Lengthening force rises smoothly to its cap. Passive force is `kpassive max(0, l-lslack)^2`.

The existing fixed tendon supplies length, speed, and moment arm. No geometry path is added. Parameters per muscle: `Fmax, lopt, width, vmax, eccentric_cap, kpassive, lslack, tau_rise, tau_fall, tau_fatigue, tau_recovery`. Positive scales and times are required.
With `GAME=1`, the game layer drains energy and stamina from active muscle work `sum(abs(Factive * tendon_speed)) * dt`. Passive tendon force does not count as muscular effort. Without muscles, the existing generalized-actuator work proxy remains in use.

## MJCF annotation

An actuator is marked with a named `<custom><numeric>` annotation `muscle_<actuator-name>`. Its eleven data values use the parameter order above. It must actuate a fixed tendon and use a control range within `[0,1]`. Unannotated actuators retain servo behavior; unsupported annotations fail export.

## Deliberate departures

This is a compact excitation-driven model with explicit activation and fatigue state, exact exponential integration, and Kinestride's existing tendon path. It does not reproduce MuJoCo's muscle gain/bias/dyntype equations or parameter names. The Python oracle defines the formula contract before the Mojo path is introduced.
