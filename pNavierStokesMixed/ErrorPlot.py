#!/usr/bin/env python3

from fenics import *
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from ConvergenceTriangles import draw_convergence_triangles, style_legend


L = 20.0
R = 0.5
PANEL_BOX_ASPECT = 0.92


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create the Section 7.3 convergence and approximation plot.'
    )
    parser.add_argument('--save-dir', type=Path, default=None, help='Optional directory for Figure 11 output.')
    parser.add_argument('--format', nargs='+', default=['pdf'], help='Output formats used with --save-dir.')
    parser.add_argument('--output-name', default='Figure11', help='Base name for saved output files.')
    parser.add_argument('--no-show', action='store_true', help='Save without opening an interactive window.')
    parser.add_argument('--no-tex', action='store_true', help='Disable LaTeX text rendering.')
    return parser.parse_args()


def configure_matplotlib(use_tex):
    plt.style.use('bmh')
    mpl.rcParams['text.usetex'] = use_tex
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['hatch.linewidth'] = 4


args = parse_args()
configure_matplotlib(use_tex=not args.no_tex)

with open('./errors.pkl', 'rb') as datei:
    dictionary = pkl.load(datei)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[1].loglog(
    dictionary['th'],
    dictionary['L2'],
    label=r'$\|\mathbf{v}_h^\tau-v_h^\tau\mathbf{e}_1\|_{L^\infty(I;L^2(\omega))}$',
    c='tab:purple',
    marker='o',
    lw=1.25,
    markersize=10.0,
    markeredgewidth=1.0,
    markerfacecolor='tab:purple',
    markeredgecolor='k',
)
ax[1].loglog(
    dictionary['th'],
    dictionary['H1'],
    label=r'$\|\mathbf{F}(\cdot,\mathbf{D}\mathbf{v}_h^\tau)-\mathbf{F}(\cdot,\mathbf{D}(v_h^\tau\mathbf{e}_1))\|_{I\times\omega}$',
    c='tab:blue',
    marker='s',
    lw=1.25,
    markersize=10.0,
    markeredgewidth=1.0,
    markerfacecolor='tab:blue',
    markeredgecolor='k',
)
ax[1].loglog(
    dictionary['th'],
    dictionary['Gamma'],
    label=r'$\|(\varphi_{\vert\mathbf{D}\mathbf{v}_h^\tau\vert+\vert\mathbf{D}(v_h^\tau\mathbf{e}_1)\vert})^*(\cdot,\vert\overline{\pi_h^\tau-\Gamma^\tau x_1}^{\,\omega}\vert)\|_{1,I\times\omega}^{1/2}$',
    c='tab:green',
    marker='^',
    lw=1.25,
    markersize=10.0,
    markeredgewidth=1.0,
    markerfacecolor='tab:green',
    markeredgecolor='k',
)

ax[1].set_axisbelow(True)
ax[1].set_facecolor('#eeeeee')
ax[1].grid(True, which='major', ls='-', color='0.75')
ax[1].grid(True, which='minor', ls=':', color='0.75')
ax[1].set_ylabel(r'Errors', fontsize=20)
ax[1].set_xlabel(r'$\tau+h$', fontsize=20)
ax[1].tick_params(axis='both', labelsize=20)

draw_convergence_triangles(ax[1], dictionary['th'], [
    {'values': dictionary['L2'], 'color': 'tab:purple', 'rate': 3.0, 'side': 'below', 'start_index': 3},
    {'values': dictionary['H1'], 'color': 'tab:blue', 'rate': 2.0},
    {'values': dictionary['Gamma'], 'color': 'tab:green', 'rate': 2.0},
], face_alpha=0.58, label_fontsize=5.0, max_height_factor=64.0)
style_legend(ax[1].legend(
    fontsize=14,
    loc='lower right',
    borderpad=0.35,
    labelspacing=0.45,
    handlelength=1.6,
    handletextpad=0.45,
))

ax[0].set_xlabel(r'$x$', fontsize=20)
ax[0].set_ylabel(
    r'$\alpha(L),\smash{v_{h_i}^{\tau_i}(L,\frac{x_{\max}}{2},x)},'
    r'\smash{\mathbf{v}_{h_i}^{\tau_i}(L,\frac{x_{\max}}{2},x)\cdot\mathbf{e}_1}$, $i=1,\ldots,8$',
    fontsize=18,
)

for n in range(1, 9):
    mesh1D = IntervalMesh(2 ** n, -R, R)
    mesh2D = RectangleMesh(Point(0.0, -R), Point(L, R), 2 ** n, 2 ** n)
    V = VectorFunctionSpace(mesh2D, 'P', 2)
    vL = Function(V, './Comparison/v' + str(n) + '/0.xml')
    X0 = mesh1D.coordinates()
    Y0 = [vL([10.0, xi])[0] for xi in X0]
    if n < 8:
        ax[0].plot(X0, Y0, c='tab:blue', ls='--', lw=1.5, alpha=0.4 + n * 0.05, marker='.', markersize=9.0 / n)
    else:
        ax[0].plot(
            X0,
            Y0,
            c='tab:blue',
            ls='-',
            lw=2.0,
            alpha=1.0,
            marker='.',
            markersize=9.0 / n,
            label=r'$\smash{\mathbf{v}_{h_i}^{\tau_i}(L,\frac{x_{\max}}{2},\cdot)\cdot\mathbf{e}_1}$, $i=1,\ldots,8$',
        )

for n in range(1, 9):
    mesh1D = IntervalMesh(2 ** n, -R, R)
    V = FunctionSpace(mesh1D, 'P', 2)
    vL = Function(V, './Comparison1D/v' + str(n) + '/0.xml')
    X0 = mesh1D.coordinates()
    Y0 = [vL(xi) for xi in X0]
    if n < 8:
        ax[0].plot(X0, Y0, c='tab:green', ls='--', lw=1.5, alpha=0.4 + n * 0.05, marker='.', markersize=9.0 / n)
    else:
        ax[0].plot(
            X0,
            Y0,
            c='tab:green',
            ls='-',
            lw=2.0,
            alpha=1.0,
            marker='.',
            markersize=9.0 / n,
            label=r'$\smash{v_{h_i}^{\tau_i}(L,\frac{x_{\max}}{2},\cdot)}$, $i=1,\ldots,8$',
        )

X = np.linspace(-R, R, 100)
ax[0].plot(X, 1.0 + 0.0 * X, ls='-', lw=1.5, color='tab:purple', label=r'$\alpha(L)$')
style_legend(ax[0].legend(fontsize=16.0, loc='best'))

ax2 = ax[0].twinx()
ax2.spines['right'].set_color('tab:red')
ax2.tick_params(axis='y', colors='tab:red')
ax2.yaxis.label.set_color('tab:red')
ax2.set_ylabel(r'$p(x)$', fontsize=20)
ax2.plot(X, 2.0 + X, c='tab:red', lw=1.5, ls='-')
ax2.set_ylim([1.0, 3.0])
for axis in ['top', 'bottom', 'left']:
    ax2.spines[axis].set_linewidth(1)
    ax2.spines[axis].set_color('k')
ax[0].tick_params(axis='both', labelsize=20)
ax2.tick_params(axis='both', labelsize=20)
for axis in (ax[0], ax2, ax[1]):
    axis.set_box_aspect(PANEL_BOX_ASPECT)

fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.94, wspace=0.38)

if args.save_dir is not None:
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for file_format in args.format:
        fig.savefig(args.save_dir / f'{args.output_name}.{file_format}', bbox_inches='tight', pad_inches=0.02)

if args.no_show:
    plt.close(fig)
else:
    plt.show()
