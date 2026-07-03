from fenics import *
import numpy as np


def _complex_profile(t, x, radius, omega):
    imaginary_unit = 1j
    wave_number = (1.0 + imaginary_unit) * np.sqrt(omega) / np.sqrt(2.0)
    exp_period = np.exp(2.0 * wave_number * radius)
    numerator = (
        np.exp(wave_number * (radius - x))
        + np.exp(wave_number * (radius + x))
        - exp_period
        - 1.0
    )
    prefactor = imaginary_unit * np.exp(imaginary_unit * omega * t) / (omega * (1.0 + exp_period))
    return prefactor * numerator


def _complex_profile_derivative(t, x, radius, omega):
    imaginary_unit = 1j
    wave_number = (1.0 + imaginary_unit) * np.sqrt(omega) / np.sqrt(2.0)
    exp_period = np.exp(2.0 * wave_number * radius)
    prefactor = imaginary_unit * np.exp(imaginary_unit * omega * t) / (omega * (1.0 + exp_period))
    return prefactor * wave_number * (
        np.exp(wave_number * (radius + x))
        - np.exp(wave_number * (radius - x))
    )


def _complex_mean_flow(t, radius, omega):
    imaginary_unit = 1j
    wave_number = (1.0 + imaginary_unit) * np.sqrt(omega) / np.sqrt(2.0)
    exp_period = np.exp(2.0 * wave_number * radius)
    prefactor = imaginary_unit * np.exp(imaginary_unit * omega * t) / (omega * (1.0 + exp_period))
    integral = 2.0 * (exp_period - 1.0) / wave_number - 2.0 * radius * (exp_period + 1.0)
    return prefactor * integral / (2.0 * radius)


class Ue(UserExpression):
    def __init__(self, t=0.0, omega=1.0, radius=1.0, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.omega = omega
        self.radius = radius

    def eval(self, values, x):
        values[0] = np.real(_complex_profile(self.t, x[0], self.radius, self.omega))

    def value_shape(self):
        return ()


class DUe(UserExpression):
    def __init__(self, t=0.0, omega=1.0, radius=1.0, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.omega = omega
        self.radius = radius

    def eval(self, values, x):
        values[0] = np.real(_complex_profile_derivative(self.t, x[0], self.radius, self.omega))

    def value_shape(self):
        return (1,)


class Gammae(UserExpression):
    """Pressure-gradient multiplier for the implemented weak form.

    The Womersley profile satisfies
    ``partial_t v - partial_xx v = cos(omega*t)``. The mixed weak form used here
    contains ``+ Gamma`` in the momentum equation, hence
    ``Gamma = -cos(omega*t)``.
    """

    def __init__(self, t=0.0, omega=1.0, radius=1.0, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.omega = omega
        self.radius = radius

    def eval(self, values, x):
        values[0] = -np.cos(self.omega * self.t)

    def value_shape(self):
        return ()


class Alphae(UserExpression):
    """Mean flow rate used in the Section 7.1 plots.

    The solver integrates ``alpha * eta`` over the cross-section. Therefore this
    normalized value enforces the total flux of the exact profile while keeping
    the plotted alpha line on the same scale as the velocity profile, as in
    Figure 5.
    """

    def __init__(self, t=0.0, omega=1.0, radius=1.0, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.omega = omega
        self.radius = radius

    def eval(self, values, x):
        values[0] = np.real(_complex_mean_flow(self.t, self.radius, self.omega))

    def value_shape(self):
        return ()
