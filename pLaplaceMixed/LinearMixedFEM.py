from fenics import *
import numpy as np

from Elements import *
from Operators import FluxConstraint


def solve_linear_mixed(mesh, T, num_steps,
                       l=1,
                       alpha=Constant(0.0),
                       uD=Constant(0.0),
                       u0=Constant(0.0),
                       snapshots=False,
                       track_point=None,
                       info=False):

    dt = T / num_steps
    W = StableElement(mesh, l=l)
    V = FunctionSpace(mesh, 'P', l)

    bc = DirichletBC(W.sub(0), uD, 'on_boundary')

    w_sol = Function(W)
    u_n = Function(V)
    u_n.assign(interpolate(u0, V))

    (u, Gamma) = TrialFunctions(W)
    (v, eta) = TestFunctions(W)

    snap_list = []
    time_series = []

    if snapshots:
        snap_list = [[u_n.copy(deepcopy=True), None]]
    if track_point is not None:
        time_series = [u_n(track_point)]

    for n in range(1, num_steps + 1):
        t = n * dt
        if hasattr(alpha, 't'):
            alpha.t = t
        if hasattr(uD, 't'):
            uD.t = t

        a = (1.0 / dt) * u * v * dx(domain=mesh) \
            + dot(grad(u), grad(v)) * dx(domain=mesh) \
            + FluxConstraint(u, Gamma, v, eta, mesh)

        # In Section 7.1 alpha is stored as the cross-section mean; integrating
        # alpha * eta over Sigma enforces the total flux of the exact profile.
        L = (1.0 / dt) * u_n * v * dx(domain=mesh) \
            + alpha * eta * dx(domain=mesh)

        solve(a == L, w_sol, bc)
        u_sol, Gamma_sol = w_sol.split(deepcopy=True)
        u_n.assign(u_sol)

        if info:
            print(f'Time step {n:4d}/{num_steps}')
        if snapshots:
            snap_list.append([u_sol.copy(deepcopy=True), Gamma_sol.copy(deepcopy=True)])
        if track_point is not None:
            time_series.append(u_n(track_point))

    return u_n, snap_list, time_series
