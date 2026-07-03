#!/usr/bin/env python3

import argparse
import pickle as pkl
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.transforms import Bbox

from ConvergenceTriangles import draw_convergence_triangles, style_legend


ERROR_LABEL_L2 = r'$\|v_h^\tau-\mathrm{I}_\tau^0v\|_{L^\infty(I;L^2(\Sigma))}$'
ERROR_LABEL_H1_LINEAR = r'$\|\nabla v_h^\tau-\nabla \mathrm{I}_\tau^0v\|_{I\times\Sigma}$'
ERROR_LABEL_H1_NONLINEAR = (
    r'$\|\mathbf{f}(\cdot,\nabla v_h^\tau)'
    r'-\mathbf{f}(\cdot,\nabla \mathrm{I}_\tau^0v)\|_{I\times\Sigma}$'
)
ERROR_LABEL_GAMMA_LINEAR = r'$\| \Gamma^\tau-\mathrm{I}_\tau^0\Gamma\|_{I}$'
ERROR_LABEL_GAMMA_MODULAR_SQRT = (
    r'$\|(\varphi_{\vert \nabla v\vert})^*(\cdot,'
    r'\vert \Gamma^\tau-\mathrm{I}_\tau^0\Gamma\vert)\|_{1,I\times\Sigma}^{1/2}$'
)
PANEL_BOX_ASPECT = 0.92


def configure_matplotlib(use_tex):
    mpl.rcParams['text.usetex'] = use_tex
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['hatch.linewidth'] = 4
    plt.style.use('bmh')


def load_errors(data_dir, output_dir):
    error_path = data_dir / output_dir / 'errors.pkl'
    if not error_path.exists():
        raise FileNotFoundError(
            f'Missing {error_path}. Regenerate the corresponding numerical data first, '
            'for example with "python3 ../run_all_sections.py --section 71 --no-plots" '
            'and/or "python3 ../run_all_sections.py --section 72 --no-plots".'
        )
    with open(error_path, 'rb') as handle:
        dictionary = pkl.load(handle)
    return {
        key: np.asarray(value, dtype=float) if isinstance(value, (list, tuple, np.ndarray)) else value
        for key, value in dictionary.items()
    }


def read_interval_function(xml_path, left, right):
    root = ET.parse(xml_path).getroot()
    dofs = [element for element in root.iter() if element.tag.endswith('dof')]
    if not dofs:
        return np.array([]), np.array([])

    num_cells = len(dofs) - 1
    h = (right - left) / num_cells
    coordinates = []
    values = []
    for dof in dofs:
        cell_index = int(dof.attrib['cell_index'])
        cell_dof_index = int(dof.attrib['cell_dof_index'])
        coordinates.append(left + (cell_index + cell_dof_index) * h)
        values.append(float(dof.attrib['value']))

    order = np.argsort(coordinates)
    return np.asarray(coordinates)[order], np.asarray(values)[order]


def trapz(values, points):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(values, points)
    return np.trapz(values, points)


def symmetric_profile(points, ps, ts):
    points = np.asarray(points, dtype=float)
    y_abs = np.abs(points)
    ps = np.asarray(ps, dtype=float)
    ts = np.asarray(ts, dtype=float)
    ps_prime = ps / (ps - 1.0)

    constants = [-1.0 / ps_prime[0] * (-ts[0]) ** ps_prime[0]]
    for index in range(1, len(ps)):
        constants.append(
            1.0 / ps_prime[index - 1] * ((-ts[index]) ** ps_prime[index - 1])
            + constants[-1]
            - 1.0 / ps_prime[index] * ((-ts[index]) ** ps_prime[index])
        )

    values = np.zeros_like(points)
    for index in range(1, len(ts)):
        mask = (y_abs >= abs(ts[index])) & (y_abs < abs(ts[index - 1]))
        values[mask] = -(
            y_abs[mask] ** ps_prime[index - 1] / ps_prime[index - 1]
            + constants[index - 1]
        )
    values[np.isclose(y_abs, abs(ts[0]))] = 0.0
    return values


def symmetric_alpha(ps, ts):
    points = np.linspace(-1.0, 1.0, 6001)
    return float(trapz(symmetric_profile(points, ps, ts), points))


def find_scalar_root(function, left, right, samples=801, tolerance=1e-13, max_iter=120):
    grid = np.linspace(left, right, samples)
    values = np.asarray([function(point) for point in grid], dtype=float)

    exact_index = int(np.argmin(np.abs(values)))
    if abs(values[exact_index]) < tolerance:
        return float(grid[exact_index])

    brackets = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if len(brackets) == 0:
        return float(grid[exact_index])

    a = float(grid[brackets[0]])
    b = float(grid[brackets[0] + 1])
    fa = float(function(a))
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = float(function(mid))
        if abs(fm) < tolerance or abs(b - a) < tolerance:
            return mid
        if fa * fm <= 0.0:
            b = mid
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)


def nonsymmetric_profile(points, p1=2.5, p2=1.5, radius=1.0, xi=0.5):
    points = np.asarray(points, dtype=float)
    p1_prime = p1 / (p1 - 1.0)
    p2_prime = p2 / (p2 - 1.0)

    def equation(shift):
        return (
            -abs(xi - shift) ** p1_prime / p1_prime
            + abs(radius + shift) ** p1_prime / p1_prime
            + abs(xi - shift) ** p2_prime / p2_prime
            - abs(radius - shift) ** p2_prime / p2_prime
        )

    shift = find_scalar_root(equation, -radius, radius)
    c1 = abs(radius + shift) ** p1_prime / p1_prime
    c2 = abs(radius - shift) ** p2_prime / p2_prime

    values = np.empty_like(points)
    lower = points <= xi
    values[lower] = -np.abs(points[lower] - shift) ** p1_prime / p1_prime + c1
    values[~lower] = -np.abs(points[~lower] - shift) ** p2_prime / p2_prime + c2
    return values


def nonsymmetric_alpha(p1=2.5, p2=1.5, radius=1.0, xi=0.5):
    points = np.linspace(-radius, radius, 6001)
    return float(trapz(nonsymmetric_profile(points, p1, p2, radius, xi), points))


def womersley_profile(points, radius, t=2.0 * np.pi, omega=1.0):
    points = np.asarray(points, dtype=float)
    imaginary_unit = 1j
    wave_number = (1.0 + imaginary_unit) * np.sqrt(omega) / np.sqrt(2.0)
    exp_period = np.exp(2.0 * wave_number * radius)
    numerator = (
        np.exp(wave_number * (radius - points))
        + np.exp(wave_number * (radius + points))
        - exp_period
        - 1.0
    )
    prefactor = imaginary_unit * np.exp(imaginary_unit * omega * t) / (omega * (1.0 + exp_period))
    return np.real(prefactor * numerator)


def womersley_alpha(radius, t=2.0 * np.pi, omega=1.0):
    return float(womersley_alpha_values(t, radius, omega))


def womersley_alpha_values(times, radius, omega=1.0):
    times = np.asarray(times, dtype=float)
    imaginary_unit = 1j
    wave_number = (1.0 + imaginary_unit) * np.sqrt(omega) / np.sqrt(2.0)
    exp_period = np.exp(2.0 * wave_number * radius)
    prefactor = imaginary_unit * np.exp(imaginary_unit * omega * times) / (omega * (1.0 + exp_period))
    integral = 2.0 * (exp_period - 1.0) / wave_number - 2.0 * radius * (exp_period + 1.0)
    return np.real(prefactor * integral / (2.0 * radius))


def style_left_axes(ax, ax_p, p_ylim=(1.0, 3.0)):
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax_p.spines['right'].set_color('tab:red')
    ax_p.tick_params(axis='y', colors='tab:red')
    ax_p.yaxis.label.set_color('tab:red')
    ax_p.set_ylabel(r'$p(x)$', fontsize=20)
    ax_p.set_ylim(p_ylim)
    for axis in ['top', 'bottom', 'left']:
        ax_p.spines[axis].set_linewidth(1)
        ax_p.spines[axis].set_color('k')
    ax_p.tick_params(axis='both', labelsize=20)


def plot_piecewise_constant(ax, intervals, color='tab:red', lw=1.5):
    for left, right, value in intervals:
        ax.plot([left, right], [value, value], c=color, lw=lw, ls='-')


def plot_approximations(ax, data_dir, output_dir, left, right, count):
    for n in range(1, count + 1):
        x_nodes, y_nodes = read_interval_function(data_dir / output_dir / f'v{n}.xml', left, right)
        alpha = 0.45 + n * (0.44 / max(count, 1))
        label = None
        if n == count:
            label = rf'$v_{{h_i}}^{{\tau_i}}(L)$, $i=1,\ldots,{count}$'
        ax.plot(
            x_nodes,
            y_nodes,
            c='tab:blue',
            ls='--',
            lw=1.5,
            alpha=alpha,
            marker='.',
            markersize=9.0 / n,
            label=label,
        )


def plot_solution_panel(ax, data_dir, output_dir, kind, count, radius=1.0, ps=None, ts=None):
    x_values = np.linspace(-radius, radius, 1000)

    if kind == 'womersley':
        u_values = womersley_profile(x_values, radius)
        alpha_value = womersley_alpha(radius)
        p_intervals = [(-radius, radius, 2.0)]
        p_ylim = (1.0, 3.0)
        ylabel = rf'$v(L,x),\alpha(L),v_{{h_i}}^{{\tau_i}}(L,x)$, $i=1,\ldots,{count}$'
    elif kind == 'symmetric':
        u_values = symmetric_profile(x_values, ps, ts)
        alpha_value = symmetric_alpha(ps, ts)
        if len(ps) == 2 and abs(ps[0] - ps[1]) < 1e-14:
            p_intervals = [(-1.0, 1.0, ps[0])]
        else:
            p_intervals = [(-1.0, -0.5, ps[0]), (-0.5, 0.5, ps[1]), (0.5, 1.0, ps[0])]
        p_ylim = (1.0, 3.0)
        ylabel = rf'$v(x),\alpha,v_{{h_i}}^{{\tau_i}}(L,x)$, $i=1,\ldots,{count}$'
    elif kind == 'nonsymmetric':
        u_values = nonsymmetric_profile(x_values)
        alpha_value = nonsymmetric_alpha()
        p_intervals = [(-1.0, 0.5, 2.5), (0.5, 1.0, 1.5)]
        p_ylim = (1.0, 3.0)
        ylabel = rf'$v(x),\alpha,v_{{h_i}}^{{\tau_i}}(L,x)$, $i=1,\ldots,{count}$'
    else:
        raise ValueError(f'Unknown solution kind: {kind}')

    alpha_values = np.full_like(x_values, alpha_value, dtype=float)
    ax.plot(x_values, alpha_values, c='tab:purple', lw=1.5, label=r'$\alpha$' if kind != 'womersley' else r'$\alpha(L)$')
    ax.plot(x_values, u_values, c='tab:blue', lw=1.5, label=r'$v$' if kind != 'womersley' else r'$v(L)$')
    ax.set_ylabel(ylabel, fontsize=20)
    plot_approximations(ax, data_dir, output_dir, -radius, radius, count)
    ax.legend(fontsize=16.0, loc='best')

    ax_p = ax.twinx()
    plot_piecewise_constant(ax_p, p_intervals)
    style_left_axes(ax, ax_p, p_ylim=p_ylim)
    return ax_p


def plot_error_panel(ax, dictionary, nonlinear, triangle_alpha):
    ax_iter = ax.twinx()
    ax.set_zorder(2)
    ax_iter.set_zorder(1)
    ax.patch.set_visible(False)
    ax_iter.patch.set_visible(True)
    ax_iter.patch.set_alpha(1.0)
    ax_iter.set_facecolor('#eeeeee')
    ax_iter.tick_params(axis='both', labelsize=20)

    gamma_values = np.sqrt(dictionary['Gamma']) if nonlinear else dictionary['Gamma']
    h1_label = ERROR_LABEL_H1_NONLINEAR if nonlinear else ERROR_LABEL_H1_LINEAR
    gamma_label = ERROR_LABEL_GAMMA_MODULAR_SQRT if nonlinear else ERROR_LABEL_GAMMA_LINEAR

    ax.loglog(
        dictionary['th'],
        dictionary['L2'],
        label=ERROR_LABEL_L2,
        c='tab:purple',
        marker='o',
        lw=1.25,
        markersize=10.0,
        markeredgewidth=1.0,
        markerfacecolor='tab:purple',
        markeredgecolor='k',
    )
    ax.loglog(
        dictionary['th'],
        dictionary['H1'],
        label=h1_label,
        c='tab:blue',
        marker='s',
        lw=1.25,
        markersize=10.0,
        markeredgewidth=1.0,
        markerfacecolor='tab:blue',
        markeredgecolor='k',
    )
    ax.loglog(
        dictionary['th'],
        gamma_values,
        label=gamma_label,
        c='tab:green',
        marker='^',
        lw=1.25,
        markersize=10.0,
        markeredgewidth=1.0,
        markerfacecolor='tab:green',
        markeredgecolor='k',
    )
    ax.set_axisbelow(True)
    ax.set_facecolor('#eeeeee')
    ax.grid(True, which='major', ls='-', color='0.75')
    ax.grid(True, which='minor', ls=':', color='0.75')
    ax.set_ylabel(r'Errors', fontsize=20)
    ax.set_xlabel(r'$\tau+h$', fontsize=20)
    draw_convergence_triangles(
        ax,
        dictionary['th'],
        [
            {'values': dictionary['L2'], 'color': 'tab:purple'},
            {'values': dictionary['H1'], 'color': 'tab:blue'},
            {'values': gamma_values, 'color': 'tab:green'},
        ],
        face_alpha=triangle_alpha,
    )

    ax_iter.semilogx(
        dictionary['th'],
        dictionary['It'],
        label=r'Number of Picard-Iterations',
        c='tab:red',
        marker='*',
        lw=1.25,
        markersize=14.0,
        markeredgewidth=1.0,
        markerfacecolor='tab:red',
        markeredgecolor='k',
        zorder=0.5,
    )
    ax_iter.set_yticks(range(1, int(np.amax(dictionary['It'])) + 1))
    ax_iter.set_axisbelow(True)
    ax_iter.set_facecolor('#eeeeee')
    ax_iter.set_ylabel(r'Number of Picard-Iterations', fontsize=20, color='tab:red')
    ax_iter.spines['right'].set_color('tab:red')
    for axis in ['top', 'bottom', 'left']:
        ax_iter.spines[axis].set_linewidth(1)
        ax_iter.spines[axis].set_color('k')
    ax_iter.yaxis.label.set_color('tab:red')
    ax_iter.tick_params(axis='y', colors='tab:red')
    ax_iter.set_xlabel(r'$\tau+h$', fontsize=20)
    style_legend(ax.legend(fontsize=16, loc='lower right'))
    ax.tick_params(axis='both', labelsize=20)
    return ax_iter


def hide_x_axis_label(*axes):
    for axis in axes:
        axis.set_xlabel('')
        axis.xaxis.label.set_visible(False)


def set_panel_box_aspect(*axes):
    for axis in axes:
        axis.set_box_aspect(PANEL_BOX_ASPECT)


def plot_case_row(axes, data_dir, output_dir, solution_kind, nonlinear, triangle_alpha, **solution_kwargs):
    dictionary = load_errors(data_dir, output_dir)
    count = len(dictionary['th'])
    ax_profile_p = plot_solution_panel(axes[0], data_dir, output_dir, solution_kind, count, **solution_kwargs)
    ax_error_iter = plot_error_panel(axes[1], dictionary, nonlinear=nonlinear, triangle_alpha=triangle_alpha)
    set_panel_box_aspect(axes[0], ax_profile_p, axes[1], ax_error_iter)
    return ax_profile_p, ax_error_iter


def make_figure5(data_dir, triangle_alpha):
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for row, radius in enumerate([1.0, 5.0, 10.0]):
        output_dir = f'Womersley{int(radius)}'
        ax_profile_p, ax_error_iter = plot_case_row(
            axes[row],
            data_dir,
            output_dir,
            'womersley',
            nonlinear=False,
            triangle_alpha=triangle_alpha,
            radius=radius,
        )
        if row < 2:
            hide_x_axis_label(axes[row, 0], axes[row, 1], ax_profile_p, ax_error_iter)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.06, top=0.97, wspace=0.36, hspace=0.06)
    return fig


def plot_womersley_surface_panel(ax, radius, omega=1.0):
    length = 2.0 * np.pi
    height = radius

    t_values = np.linspace(0.0, length, 200)
    x_values = np.linspace(-height, height, 200)
    t_grid, x_grid = np.meshgrid(t_values, x_values)
    velocity = womersley_profile(x_grid, radius, t=t_grid, omega=omega)
    alpha = np.broadcast_to(
        womersley_alpha_values(t_grid, radius, omega=omega),
        velocity.shape,
    )
    zmax = np.amax(velocity)

    light_source = LightSource(azdeg=315, altdeg=45)
    rgb = light_source.shade(velocity, cmap=cm.viridis, vert_exag=0.1, blend_mode='soft')
    ax.plot_surface(
        t_grid,
        x_grid,
        velocity,
        rstride=1,
        cstride=1,
        facecolors=rgb,
        antialiased=True,
        zorder=1,
    )

    alpha_above = np.ma.masked_where(alpha <= velocity, alpha)
    alpha_below = np.ma.masked_where(alpha > velocity, alpha)
    ax.plot_surface(
        t_grid,
        x_grid,
        alpha_above,
        rstride=1,
        cstride=1,
        color='tab:purple',
        alpha=0.25,
        antialiased=True,
        zorder=0,
    )
    ax.plot_surface(
        t_grid,
        x_grid,
        alpha_below,
        rstride=1,
        cstride=1,
        color='tab:purple',
        alpha=0.25,
        antialiased=True,
        zorder=2,
    )

    alpha_line = womersley_alpha_values(t_values, radius, omega=omega)
    ax.plot(t_values, 0.0 * t_values - radius, alpha_line, lw=0.75, c='tab:purple', zorder=5)
    ax.plot(t_values, 0.0 * t_values + radius, alpha_line, lw=0.75, c='tab:purple', ls='dashed', zorder=0)

    ax.plot([0.0, length], [radius, radius], [0.0, 0.0], color='tab:gray', ls='dashed', lw=0.5, zorder=0)
    ax.plot([0.0, length], [-radius, -radius], [-zmax, -zmax], color='tab:gray', ls='dashed', lw=0.5, zorder=10)
    ax.plot([0.0, length], [radius, radius], [-zmax, -zmax], color='tab:gray', ls='dashed', lw=0.5, zorder=0)
    ax.plot([length, length], [radius, radius], [-zmax, zmax], color='tab:gray', ls='dashed', lw=0.5, zorder=0)

    ax.set_yticks([-radius, -radius / 2.0, 0.0, radius / 2.0, radius])
    ax.set_zticks([
        -int(10.0 * zmax) / 10.0,
        -int(10.0 * zmax) / 20.0,
        0.0,
        int(10.0 * zmax) / 20.0,
        int(10.0 * zmax) / 10.0,
    ])

    ax.set_xlabel(r'$t$', labelpad=8, fontsize=8, fontname='Times New Roman')
    ax.set_ylabel(r'$x$', labelpad=0, fontsize=8, fontname='Times New Roman')
    ax.set_zlabel(r'$v(t,x),\alpha(t)$', labelpad=0, fontsize=8, fontname='Times New Roman')
    try:
        ax.set_box_aspect(aspect=(4, 1, 1))
    except TypeError:
        ax.set_box_aspect(aspect=(4, 1, 1))

    ax.plot([0.0, length], [-radius, -radius], [0.0, 0.0], color='tab:gray', lw=0.5, zorder=10)
    ax.plot([length, length], [-radius, -radius], [-zmax, zmax], color='tab:gray', lw=0.5, zorder=10)
    ax.plot([0.0, 0.0], [-radius, -radius], [-zmax, zmax], color='tab:gray', lw=0.5, zorder=10)
    ax.plot([0.0, 0.0], [radius, radius], [-zmax, zmax], color='tab:gray', lw=0.5, zorder=0)

    for ztick in [-zmax, -zmax / 2.0, 0.0, zmax / 2.0, zmax]:
        ax.plot([0.0, 0.0], [-radius, radius], [ztick, ztick], color='tab:gray', ls='dashed', lw=0.5, zorder=0)
        ax.plot([length, length], [-radius, radius], [ztick, ztick], color='tab:gray', ls='dashed', lw=0.5, zorder=10)

    for ttick in range(7):
        ax.plot([ttick, ttick], [-radius, -radius], [-zmax, 0.0], color='tab:gray', ls='dashed', lw=0.5, zorder=10)
        ax.plot([ttick, ttick], [radius, radius], [-zmax, 0.0], color='tab:gray', ls='dashed', lw=0.5, zorder=0)

    ax.tick_params(axis='both', labelsize=8)


def make_figure6(data_dir, triangle_alpha):
    use_tex = mpl.rcParams['text.usetex']
    with plt.style.context('default'):
        mpl.rcParams['text.usetex'] = use_tex
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{color}'
        mpl.rcParams['font.family'] = 'serif'

        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_alpha(0.0)
        fig._save_transparent = True
        fig._save_bbox_inches = Bbox.from_extents(3.25, 0.50, 9.75, 7.35)
        fig._save_pad_inches = 0.0
        panel_positions = [
            [0.00, 0.57, 0.95, 0.39],
            [0.00, 0.31, 0.95, 0.39],
            [0.00, 0.05, 0.95, 0.39],
        ]
        for panel_position, radius in zip(panel_positions, [1.0, 5.0, 10.0]):
            ax = fig.add_axes(panel_position, projection='3d')
            plot_womersley_surface_panel(ax, radius=radius, omega=1.0)
        return fig


def make_figure7(data_dir, triangle_alpha):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.8))
    cases = [
        ('Const25', [2.5, 2.5]),
        ('Const15', [1.5, 1.5]),
    ]
    for row, (output_dir, ps) in enumerate(cases):
        ax_profile_p, ax_error_iter = plot_case_row(
            axes[row],
            data_dir,
            output_dir,
            'symmetric',
            nonlinear=True,
            triangle_alpha=triangle_alpha,
            radius=1.0,
            ps=ps,
            ts=[-1.0, -0.5, 0.0],
        )
        if row == 0:
            hide_x_axis_label(axes[row, 0], axes[row, 1], ax_profile_p, ax_error_iter)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.09, top=0.96, wspace=0.38, hspace=0.06)
    return fig


def make_figure8(data_dir, triangle_alpha):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_case_row(
        axes,
        data_dir,
        'Sym1',
        'symmetric',
        nonlinear=True,
        triangle_alpha=triangle_alpha,
        radius=1.0,
        ps=[1.5, 2.5],
        ts=[-1.0, -0.5, 0.0],
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.94, wspace=0.38)
    return fig


def make_figure9(data_dir, triangle_alpha):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_case_row(
        axes,
        data_dir,
        'NonSym2',
        'nonsymmetric',
        nonlinear=True,
        triangle_alpha=triangle_alpha,
        radius=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.94, wspace=0.38)
    return fig


def save_figure(fig, save_dir, stem, formats):
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_format in formats:
        path = save_dir / f'{stem}.{file_format}'
        fig.savefig(
            path,
            bbox_inches=getattr(fig, '_save_bbox_inches', 'tight'),
            pad_inches=getattr(fig, '_save_pad_inches', 0.03),
            transparent=getattr(fig, '_save_transparent', False),
        )
        paths.append(path)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Create paper Figures 5, 6, 7, 8 and 9 from stored data.')
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='pLaplaceMixed directory with errors.pkl files.')
    parser.add_argument('--figure', choices=['all', '5', '6', '7', '8', '9'], default='all')
    parser.add_argument('--save-dir', type=Path, default=Path('plots'), help='Directory for generated figures.')
    parser.add_argument('--format', nargs='+', default=['pdf', 'png'], help='Output formats, e.g. pdf png.')
    parser.add_argument('--no-show', action='store_true', help='Do not open plot windows.')
    parser.add_argument('--no-tex', action='store_true', help='Disable LaTeX text rendering.')
    parser.add_argument('--triangle-alpha', type=float, default=0.58, help='Opacity of convergence triangles.')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_matplotlib(use_tex=not args.no_tex)
    makers = {
        '5': make_figure5,
        '6': make_figure6,
        '7': make_figure7,
        '8': make_figure8,
        '9': make_figure9,
    }
    selected = ['5', '6', '7', '8', '9'] if args.figure == 'all' else [args.figure]
    for figure_number in selected:
        fig = makers[figure_number](args.data_dir, args.triangle_alpha)
        paths = save_figure(fig, args.save_dir, f'Figure{figure_number}', args.format)
        print('\n'.join(str(path) for path in paths))
        if args.no_show:
            plt.close(fig)
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
