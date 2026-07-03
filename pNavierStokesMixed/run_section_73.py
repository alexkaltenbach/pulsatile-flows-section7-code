#!/usr/bin/env python3

import argparse
import subprocess
import sys


STAGES = {
    'one-d': ['ErrorDecayExperiment1D.py'],
    'two-d': ['ErrorDecayExperiment.py'],
    'compare': ['ErrorDecayComparison.py'],
    'plot': ['ErrorPlot.py'],
    'surface': ['PlotSolution.py'],
}


def run_command(args):
    print(' '.join(args))
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run the numerical experiment for Section 7.3.'
    )
    parser.add_argument(
        '--stage',
        choices=['all', 'one-d', 'two-d', 'compare', 'plot', 'surface'],
        default='all',
        help='Select which part of the experiment to run.',
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=9,
        help='Run refinement levels n=1,...,nmax-1. Section 7.3 uses nmax=9.',
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Run computations and error comparison without opening plots.',
    )
    args = parser.parse_args()

    if args.stage == 'all':
        stages = ['one-d', 'two-d', 'compare']
        if not args.skip_plots:
            stages.extend(['plot', 'surface'])
    else:
        stages = [args.stage]

    for stage in stages:
        script_args = [sys.executable] + STAGES[stage]
        if stage in ('one-d', 'two-d', 'compare'):
            script_args.extend(['--nmax', str(args.nmax)])
        run_command(script_args)


if __name__ == '__main__':
    main()
