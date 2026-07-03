#!/usr/bin/env python3

from fenics import *
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource, ListedColormap, Normalize, PowerNorm


L = 20.0
R = 0.5
EPS = L / 4.0


class LiftVelocityTo2D(UserExpression):
    def __init__(self, velocity_1d, **kwargs):
        self.velocity_1d = velocity_1d
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.velocity_1d(x[1])
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class LiftGammaToPressure(UserExpression):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.gamma(0.0) * x[0]

    def value_shape(self):
        return ()


def configure_matplotlib(use_tex):
    mpl.rcParams['text.usetex'] = use_tex
    mpl.rcParams['text.latex.preamble'] = (
        r'\usepackage{color}\usepackage{amsmath}\usepackage{amsthm}\usepackage{amssymb}'
    )
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.transparent'] = True


def load_snapshot_pair(n, time_index, one_d_dir, two_d_dir):
    mesh_1d = IntervalMesh(2 ** n, -R, R)
    v_1d = FunctionSpace(mesh_1d, 'P', 2)
    q_1d = FunctionSpace(mesh_1d, 'R', 0)

    mesh_2d = RectangleMesh(Point(0.0, -R), Point(L, R), 2 ** n, 2 ** n)
    v_2d = VectorFunctionSpace(mesh_2d, 'P', 2)
    q_2d = FunctionSpace(mesh_2d, 'P', 1)

    velocity_1d = Function(v_1d, str(Path(one_d_dir) / f'v{n}' / f'{time_index}.xml'))
    gamma_1d = Function(q_1d, str(Path(one_d_dir) / f'Gamma{n}' / f'{time_index}.xml'))

    velocity_1d_lift = interpolate(LiftVelocityTo2D(velocity_1d, degree=2), v_2d)
    pressure_1d_lift = interpolate(LiftGammaToPressure(gamma_1d, degree=1), q_2d)

    velocity_2d = Function(v_2d, str(Path(two_d_dir) / f'v{n}' / f'{time_index}.xml'))
    pressure_2d = Function(q_2d, str(Path(two_d_dir) / f'pi{n}' / f'{time_index}.xml'))
    return velocity_1d_lift, velocity_2d, pressure_1d_lift, pressure_2d


def pressure_difference_mean_on_omega(pressure_2d, pressure_1d, n):
    mesh_omega = RectangleMesh(Point(EPS, -R), Point(L - EPS, R), max(1, 2 ** (n - 1)), 2 ** n)
    q_omega = FunctionSpace(mesh_omega, 'P', 1)
    pressure_2d_omega = interpolate(pressure_2d, q_omega)
    pressure_1d_omega = interpolate(pressure_1d, q_omega)
    difference = pressure_2d_omega - pressure_1d_omega
    volume = assemble(Constant(1.0) * dx(domain=mesh_omega))
    return assemble(difference * dx(domain=mesh_omega)) / volume


def sample_pressure_difference(pressure_2d, pressure_1d, x_grid, y_grid, omega_mean):
    values = [
        abs((pressure_2d([xi, yi]) - pressure_1d([xi, yi])) - omega_mean)
        for xi, yi in zip(np.ravel(x_grid), np.ravel(y_grid))
    ]
    return np.asarray(values).reshape(x_grid.shape)


def sample_velocity_difference(velocity_2d, velocity_1d, x_grid, y_grid):
    values = []
    for xi, yi in zip(np.ravel(x_grid), np.ravel(y_grid)):
        difference = np.asarray(velocity_2d([xi, yi])) - np.asarray(velocity_1d([xi, yi]))
        values.append(np.linalg.norm(difference))
    return np.asarray(values).reshape(x_grid.shape)


def make_grids():
    x_full = np.linspace(0.0, L, 200)
    y_full = np.linspace(-R, R, 100)
    x_full, y_full = np.meshgrid(x_full, y_full)

    x_left = np.linspace(0.0, L / 4.0, 50)
    x_mid = np.linspace(L / 4.0, 3.0 * L / 4.0, 100)
    x_right = np.linspace(3.0 * L / 4.0, L, 50)
    y_values = np.linspace(-R, R, 100)

    x_left, y_left = np.meshgrid(x_left, y_values)
    x_mid, y_mid = np.meshgrid(x_mid, y_values)
    x_right, y_right = np.meshgrid(x_right, y_values)
    return (x_full, y_full), (x_left, y_left), (x_mid, y_mid), (x_right, y_right)


def make_norm(values):
    norm_min = float(np.amin(values))
    norm_max = float(np.amax(values))
    if norm_max <= norm_min:
        norm_max = norm_min + DOLFIN_EPS
    return PowerNorm(gamma=0.5, vmin=norm_min, vmax=norm_max)


def plot_error_surface(ax, grids, z_full, z_left, z_mid, z_right, z_label, lift_base=None):
    (_, _), (x_left, y_left), (x_mid, y_mid), (x_right, y_right) = grids
    z_max = max(float(np.amax(z_left)), float(np.amax(z_mid)), float(np.amax(z_right)))
    z_min = min(float(np.amin(z_left)), float(np.amin(z_mid)), float(np.amin(z_right)))
    norm = make_norm(z_full)

    light_source = LightSource(azdeg=315, altdeg=45)
    rgb_left = light_source.shade(norm(z_left), cmap=cm.viridis, vert_exag=0.1, blend_mode='soft', norm=Normalize(0, 1))
    rgb_mid = light_source.shade(norm(z_mid), cmap=cm.viridis, vert_exag=0.1, blend_mode='soft', norm=Normalize(0, 1))
    rgb_right = light_source.shade(norm(z_right), cmap=cm.viridis, vert_exag=0.1, blend_mode='soft', norm=Normalize(0, 1))
    rgb_mid_lifted = light_source.shade(norm(z_mid), cmap=ListedColormap(['tab:green']), vert_exag=0.1, blend_mode='soft')

    ax.plot_surface(x_left, y_left, z_left, rstride=1, cstride=1, facecolors=rgb_left, antialiased=True, zorder=1)
    ax.plot_surface(x_left, y_left, 0.0 * z_left + z_min, rstride=1, cstride=1, color='tab:blue', antialiased=True, zorder=1, alpha=0.5)
    ax.plot_surface(x_mid, y_mid, z_mid, rstride=1, cstride=1, facecolors=rgb_mid, antialiased=True, zorder=1)
    ax.plot_surface(x_right, y_right, z_right, rstride=1, cstride=1, facecolors=rgb_right, antialiased=True, zorder=1)
    ax.plot_surface(x_right, y_right, 0.0 * z_right + z_min, rstride=1, cstride=1, color='tab:blue', antialiased=True, zorder=1, alpha=0.5)

    if lift_base is None:
        lift_base = 3.5 if z_max > 3.5 else 0.05 * z_max
    lift_top = z_max
    if lift_top <= lift_base:
        lift_top = lift_base + max(0.1 * max(z_max, 1.0), DOLFIN_EPS)
    z_mid_lifted = lift_base + z_mid / max(float(np.amax(z_mid)), DOLFIN_EPS) * (lift_top - lift_base)
    ax.plot_surface(x_mid, y_mid, z_mid_lifted, rstride=1, cstride=1, facecolors=rgb_mid_lifted, antialiased=True, zorder=1)
    ax.plot_surface(x_mid, y_mid, 0.0 * z_mid + 0.05, rstride=1, cstride=1, color='tab:green', antialiased=True, zorder=2, alpha=0.75)

    x_pos = 15.0
    y_pos = R
    ax.plot([x_pos, x_pos], [y_pos, y_pos], [lift_base, lift_top + 0.01], color='tab:green', lw=1)
    z_ticks = np.linspace(lift_base, lift_top, 4)
    z_labels = (z_ticks - lift_base) / max(lift_top - lift_base, DOLFIN_EPS) * float(np.amax(z_mid))
    for tick, label in zip(z_ticks, z_labels):
        ax.plot([x_pos - 0.25, x_pos + 0.25], [y_pos, y_pos], [tick, tick], color='tab:green', lw=1)
        ax.text(x_pos + 0.75, y_pos - 0.05, tick, f'{label:.2e}', color='tab:green', fontsize=10)

    ax.set_yticks([-R, -R / 2.0, 0.0, R / 2.0, R])
    ax.set_xlabel(r'$x_1$', labelpad=8, fontsize=10, fontname='Times New Roman')
    ax.set_ylabel(r'$\overline{x}$', labelpad=0, fontsize=10, fontname='Times New Roman')
    ax.set_zlabel(z_label, labelpad=0, fontsize=10, fontname='Times New Roman')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_box_aspect(aspect=(4, 1, 1))


def make_figures(n, time_index, one_d_dir, two_d_dir):
    velocity_1d, velocity_2d, pressure_1d, pressure_2d = load_snapshot_pair(
        n,
        time_index,
        one_d_dir,
        two_d_dir,
    )
    omega_mean = pressure_difference_mean_on_omega(pressure_2d, pressure_1d, n)
    grids = make_grids()
    (x_full, y_full), (x_left, y_left), (x_mid, y_mid), (x_right, y_right) = grids

    pressure_full = sample_pressure_difference(pressure_2d, pressure_1d, x_full, y_full, omega_mean)
    pressure_left = sample_pressure_difference(pressure_2d, pressure_1d, x_left, y_left, omega_mean)
    pressure_mid = sample_pressure_difference(pressure_2d, pressure_1d, x_mid, y_mid, omega_mean)
    pressure_right = sample_pressure_difference(pressure_2d, pressure_1d, x_right, y_right, omega_mean)

    velocity_full = sample_velocity_difference(velocity_2d, velocity_1d, x_full, y_full)
    velocity_left = sample_velocity_difference(velocity_2d, velocity_1d, x_left, y_left)
    velocity_mid = sample_velocity_difference(velocity_2d, velocity_1d, x_mid, y_mid)
    velocity_right = sample_velocity_difference(velocity_2d, velocity_1d, x_right, y_right)

    pressure_fig = plt.figure(figsize=(12, 8))
    velocity_fig = plt.figure(figsize=(12, 8))
    pressure_ax = pressure_fig.add_subplot(1, 1, 1, projection='3d')
    velocity_ax = velocity_fig.add_subplot(1, 1, 1, projection='3d')

    pressure_label = (
        r'$\vert\overline{'
        rf'\Gamma^{{\tau_{n}}}(\frac{{L}}{{2}})x_1-'
        rf'\pi^{{\tau_{n}}}_{{h_{n}}}(\frac{{L}}{{2}},x_1,\overline{{x}})'
        r'}^{\,\omega}\vert$'
    )
    velocity_label = (
        rf'$\vert v^{{\tau_{n}}}_{{h_{n}}}(\frac{{L}}{{2}},x_1,\overline{{x}})\mathbf{{e}}_1-'
        rf'\mathbf{{v}}^{{\tau_{n}}}_{{h_{n}}}(\frac{{L}}{{2}},x_1,\overline{{x}})\vert$'
    )

    plot_error_surface(
        pressure_ax,
        grids,
        pressure_full,
        pressure_left,
        pressure_mid,
        pressure_right,
        pressure_label,
        lift_base=3.5,
    )
    plot_error_surface(
        velocity_ax,
        grids,
        velocity_full,
        velocity_left,
        velocity_mid,
        velocity_right,
        velocity_label,
        lift_base=0.05 * max(float(np.amax(velocity_full)), DOLFIN_EPS),
    )
    return {
        'pressure': pressure_fig,
        'velocity': velocity_fig,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Create the separate Section 7.3 Figure-12 pressure and velocity surface plots.')
    parser.add_argument('--n', type=int, default=8, help='Refinement level used for the displayed snapshot.')
    parser.add_argument(
        '--time-index',
        type=int,
        default=None,
        help='Snapshot index. Defaults to 2^(n-1), the midpoint time used in the paper plot.',
    )
    parser.add_argument('--one-d-dir', default='Comparison1D', help='Directory with reduced 1D snapshots.')
    parser.add_argument('--two-d-dir', default='Comparison', help='Directory with full 2D snapshots.')
    parser.add_argument('--save-dir', type=Path, default=None, help='Optional directory for Figure12 output.')
    parser.add_argument('--format', nargs='+', default=['pdf'], help='Output formats used with --save-dir.')
    parser.add_argument('--no-show', action='store_true', help='Save without opening an interactive window.')
    parser.add_argument('--no-tex', action='store_true', help='Disable LaTeX text rendering.')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib(use_tex=not args.no_tex)
    time_index = args.time_index if args.time_index is not None else 2 ** (args.n - 1)
    figures = make_figures(args.n, time_index, args.one_d_dir, args.two_d_dir)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            for file_format in args.format:
                output_path = args.save_dir / f'Figure12_{name}.{file_format}'
                fig.savefig(output_path)

    if args.no_show:
        for fig in figures.values():
            plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
