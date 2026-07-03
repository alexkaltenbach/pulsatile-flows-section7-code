#!/usr/bin/env python3
"""Run the numerical experiments from Section 7.1.

Examples:
    python3 run_section_71.py --radius 1 --nmax 12
    python3 run_section_71.py --radius all --nmax 12 --plot
"""

import argparse
import subprocess
import sys


RADII = [1.0, 5.0, 10.0]


def output_dir_for_radius(radius):
    return f'Womersley{int(radius) if float(radius).is_integer() else radius}'


def run_case(radius, nmax, make_plot):
    from fenics import Constant
    import numpy as np
    from ErrorDecayAnalysisWomersley import error_decay_analysis

    output_dir = output_dir_for_radius(radius)
    print(f'\n=== Section 7.1: radius r = {radius:g} -> {output_dir}/ ===')
    error_decay_analysis(
        2.0 * np.pi,
        radius=radius,
        nmax=nmax,
        l=1,
        omega=1.0,
        u0=Constant(0.0),
        output_dir=output_dir,
    )

    if make_plot:
        subprocess.run([sys.executable, 'ErrorPlotWomersley.py', '--radius', str(radius), '--output-dir', output_dir], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Section 7.1 Womersley experiments.')
    parser.add_argument(
        '--radius',
        choices=['all', '1', '5', '10'],
        default='all',
        help='Radius to run. Paper setup uses 1, 5, and 10.',
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=12,
        help='Upper refinement loop bound. The code runs n=1,...,nmax-1. Paper setup: nmax=12.',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Run the matching plotting script after computing each selected radius.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    radii = RADII if args.radius == 'all' else [float(args.radius)]
    for radius in radii:
        run_case(radius, args.nmax, args.plot)


if __name__ == '__main__':
    main()
