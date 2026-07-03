import argparse
import pickle as pkl

from fenics import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ConvergenceTriangles import draw_convergence_triangles, style_legend
from DataWomersley import Alphae, Ue


def output_dir_for_radius(radius):
    return f'Womersley{int(radius) if float(radius).is_integer() else radius}'


def parse_args():
    parser = argparse.ArgumentParser(description='Plot a Section 7.1 Womersley experiment.')
    parser.add_argument('--radius', type=float, required=True, choices=[1.0, 5.0, 10.0])
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    radius = args.radius
    output_dir = output_dir_for_radius(radius) if args.output_dir is None else args.output_dir

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['hatch.linewidth'] = 4
    plt.style.use('bmh')

    with open('./' + output_dir + '/errors.pkl', 'rb') as datei:
        dictionary = pkl.load(datei)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = ax[1].twinx()
    ax[1].set_zorder(2)
    ax1.set_zorder(1)
    ax[1].patch.set_visible(False)
    ax1.patch.set_visible(True)
    ax1.patch.set_alpha(1.0)
    ax1.set_facecolor('#eeeeee')
    ax1.tick_params(axis='both', labelsize=20)

    ax[1].loglog(dictionary['th'], dictionary['L2'], label=r'$\|v_h^\tau-\mathrm{I}_\tau^0v\|_{L^\infty(I;L^2(\Sigma))}$', c='tab:purple', marker='o', lw=1.25,
                 markersize=10.0, markeredgewidth=1.0, markerfacecolor='tab:purple', markeredgecolor='k')
    ax[1].loglog(dictionary['th'], dictionary['H1'], label=r'$\|\nabla v_h^\tau-\nabla \mathrm{I}_\tau^0v\|_{I\times\Sigma}$', c='tab:blue', marker='s', lw=1.25,
                 markersize=10.0, markeredgewidth=1.0, markerfacecolor='tab:blue', markeredgecolor='k')
    ax[1].loglog(dictionary['th'], dictionary['Gamma'], label=r'$\| \Gamma^\tau-\mathrm{I}_\tau^0\Gamma\|_{I}$', c='tab:green', marker='^', lw=1.25,
                 markersize=10.0, markeredgewidth=1.0, markerfacecolor='tab:green', markeredgecolor='k')

    ax[1].set_axisbelow(True)
    ax[1].set_facecolor('#eeeeee')
    ax[1].grid(True, which='major', ls='-', color='0.75')
    ax[1].grid(True, which='minor', ls=':', color='0.75')
    ax[1].set_ylabel(r'Errors', fontsize=20)
    ax[1].set_xlabel(r'$\tau+h$', fontsize=20)

    draw_convergence_triangles(ax[1], dictionary['th'], [
        {'values': dictionary['L2'], 'color': 'tab:purple'},
        {'values': dictionary['H1'], 'color': 'tab:blue'},
        {'values': dictionary['Gamma'], 'color': 'tab:green'},
    ])

    ax1.semilogx(dictionary['th'], dictionary['It'], label=r'Number of Picard-Iterations', c='tab:red', marker='*', lw=1.25,
                 markersize=14.0, markeredgewidth=1.0, markerfacecolor='tab:red', markeredgecolor='k', zorder=0.5)
    ax1.set_yticks(range(1, np.amax(dictionary['It']) + 1))
    ax1.set_axisbelow(True)
    ax1.set_facecolor('#eeeeee')
    ax1.set_ylabel(r'Number of Picard-Iterations', fontsize=20, color='tab:red')
    ax1.spines['right'].set_color('tab:red')
    for axis in ['top', 'bottom', 'left']:
        ax1.spines[axis].set_linewidth(1)
        ax1.spines[axis].set_color('k')
    ax1.yaxis.label.set_color('tab:red')
    ax1.tick_params(axis='y', colors='tab:red')
    ax1.set_xlabel(r'$\tau+h$', fontsize=20)
    style_legend(ax[1].legend(fontsize=16, loc='lower right'))
    ax[1].tick_params(axis='both', labelsize=20)

    ue = Ue(t=2.0 * np.pi, omega=1.0, radius=radius, degree=6)
    alphae = Alphae(t=2.0 * np.pi, omega=1.0, radius=radius, degree=6)
    x_values = np.linspace(-radius, radius, 1000)
    u_values = [ue(xi) for xi in x_values]
    alpha_values = [alphae(xi) for xi in x_values]
    p_values = [2.0 for _ in x_values]

    ax[0].plot(x_values, alpha_values, c='tab:purple', lw=1.5, label=r'$\alpha(L)$')
    ax[0].plot(x_values, u_values, c='tab:blue', lw=1.5, label=r'$v(L)$')
    ax[0].set_xlabel(r'$x$', fontsize=20)
    ax[0].set_ylabel(r'$v(L,x),\alpha(L),\smash{v_{h_i}^{\tau_i}(L,x)}$, $i=1,\ldots,11$', fontsize=20)

    for n in range(1, len(dictionary['th']) + 1):
        mesh = IntervalMesh(2 ** n, -radius, radius)
        V = FunctionSpace(mesh, 'P', 1)
        vL = Function(V, './' + output_dir + '/v' + str(n) + '.xml')
        x_mesh = mesh.coordinates()
        y_mesh = [vL(xi) for xi in x_mesh]
        if n < len(dictionary['th']):
            ax[0].plot(x_mesh, y_mesh, c='tab:blue', ls='--', lw=1.5, alpha=0.45 + n * 0.04, marker='.', markersize=9.0 / n)
        else:
            ax[0].plot(x_mesh, y_mesh, c='tab:blue', ls='--', lw=1.5, alpha=0.45 + n * 0.04, marker='.', markersize=9.0 / n,
                       label=r'$\smash{v_{h_i}^{\tau_i}(L)}$, $i=1,\ldots,11$')

    ax[0].legend(fontsize=16.0, loc='best')
    ax2 = ax[0].twinx()
    ax2.spines['right'].set_color('tab:red')
    ax2.tick_params(axis='y', colors='tab:red')
    ax2.yaxis.label.set_color('tab:red')
    ax2.set_ylabel(r'$p(x)$', fontsize=20)
    ax2.plot(x_values, p_values, c='tab:red', lw=1.5, ls='-')
    ax2.set_ylim([1.0, 3.0])
    for axis in ['top', 'bottom', 'left']:
        ax2.spines[axis].set_linewidth(1)
        ax2.spines[axis].set_color('k')
    ax[0].tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.94, wspace=0.38)
    plt.show()


if __name__ == '__main__':
    main()
