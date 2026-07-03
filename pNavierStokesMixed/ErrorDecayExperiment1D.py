#!/usr/bin/env python3
"""Generate the reduced 1D comparison data for Section 7.3."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from fenics import *


ROOT = Path(__file__).resolve().parent
PLAPLACE_ROOT = ROOT.parent / 'pLaplaceMixed'
if str(PLAPLACE_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAPLACE_ROOT))

from PicardIteration import PicardIteration


def run_error_decay_experiment_1d(
        T,
        nmax=9,
        delta=DOLFIN_EPS,
        output_dir='Comparison1D',
        R=0.5):
    """Run the reduced 1D problem associated with the full 2D Section 7.3 setup."""

    alpha = Expression('cos(t)', degree=6, t=0.0)
    exponent = Expression('2.0 + x[0]', degree=1)
    stress_factor = Expression('pow(2.0, -0.5 * (2.0 + x[0]))', degree=1)
    dictionary = {'It': [], 'nmax': nmax, 'levels': list(range(1, nmax))}
    os.makedirs(output_dir, exist_ok=True)

    for n in range(1, nmax):
        print('Step:', n)
        mesh = IntervalMesh(2 ** n, -R, R)
        num_steps = 2 ** n

        snap_list, _, _, num_iter = PicardIteration(
            mesh,
            T,
            num_steps,
            l=2,
            alpha=alpha,
            delta=delta,
            r=exponent,
            stress_factor=stress_factor,
            u0=Constant(0.0),
            snapshots=True,
            info=True,
            inner_info=False,
        )

        os.makedirs(output_dir + '/v' + str(n), exist_ok=True)
        os.makedirs(output_dir + '/Gamma' + str(n), exist_ok=True)
        for i, (velocity, gamma) in enumerate(snap_list):
            File(output_dir + '/v' + str(n) + '/' + str(i) + '.xml') << velocity
            if gamma is not None:
                File(output_dir + '/Gamma' + str(n) + '/' + str(i) + '.xml') << gamma

        dictionary['It'].append(num_iter)

    with open(output_dir + '/iterations.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the reduced 1D Section 7.3 comparison experiment.')
    parser.add_argument('--nmax', type=int, default=9, help='Upper refinement loop bound. Runs n=1,...,nmax-1; paper setup: nmax=9.')
    parser.add_argument('--output-dir', default='Comparison1D', help='Directory for 1D snapshots.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_error_decay_experiment_1d(2.0 * np.pi, nmax=args.nmax, output_dir=args.output_dir)
