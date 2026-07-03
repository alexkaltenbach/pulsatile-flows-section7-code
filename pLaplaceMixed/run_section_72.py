#!/usr/bin/env python3
"""Run the numerical experiments from Section 7.2.

Examples:
    python3 run_section_72.py --case const15 --nmax 10
    python3 run_section_72.py --case even --nmax 6
    python3 run_section_72.py --case all --plot
"""

import argparse
import subprocess
import sys


CASES = {
    'const15': {
        'description': 'Constant case p = 3/2, alpha = 1/2, formula (7.3)',
        'output_dir': 'Const15',
        'plot_script': 'ErrorPlotConst15.py',
        'ps': [ 1.5, 1.5 ],
        'type': 'symmetric',
    },
    'const25': {
        'description': 'Constant case p = 5/2, alpha = 3/4, formula (7.3)',
        'output_dir': 'Const25',
        'plot_script': 'ErrorPlotConst25.py',
        'ps': [ 2.5, 2.5 ],
        'type': 'symmetric',
    },
    'even': {
        'description': 'Even piecewise-constant case, formula (7.4)',
        'output_dir': 'Sym1',
        'plot_script': 'ErrorPlotSym.py',
        'ps': [ 1.5, 2.5 ],
        'type': 'symmetric',
    },
    'noneven': {
        'description': 'Non-even piecewise-constant case, formula (7.5)',
        'output_dir': 'NonSym2',
        'plot_script': 'ErrorPlotNonSym.py',
        'type': 'nonsymmetric',
    },
}


def run_case( case_name, nmax, make_plot ):
    from fenics import Constant, DOLFIN_EPS
    from ErrorDecayAnalysisNonSym import error_decay_analysis as run_nonsymmetric_case
    from ErrorDecayAnalysisSym import error_decay_analysis as run_symmetric_case

    case = CASES[ case_name ]
    print( '\n=== ' + case_name + ': ' + case[ 'description' ] + ' ===' )

    if case[ 'type' ] == 'symmetric':
        run_symmetric_case(
            1.0,
            nmax = nmax,
            l = 1,
            delta = DOLFIN_EPS,
            ps = case[ 'ps' ],
            filename = case[ 'output_dir' ],
            u0 = Constant( 0.0 ),
        )
    else:
        run_nonsymmetric_case(
            1.0,
            nmax = nmax,
            l = 1,
            delta = DOLFIN_EPS,
            u0 = Constant( 0.0 ),
            output_dir = case[ 'output_dir' ],
        )

    if make_plot:
        subprocess.run( [ sys.executable, case[ 'plot_script' ] ], check = True )


def parse_args():
    parser = argparse.ArgumentParser( description = 'Run Section 7.2 experiments.' )
    parser.add_argument(
        '--case',
        choices = [ 'all' ] + sorted( CASES.keys() ),
        default = 'all',
        help = 'Experiment to run. Use nmax=10 for the full paper setup.',
    )
    parser.add_argument(
        '--nmax',
        type = int,
        default = 10,
        help = 'Upper refinement loop bound. The code runs n=1,...,nmax-1. Paper setup: nmax=10.',
    )
    parser.add_argument(
        '--plot',
        action = 'store_true',
        help = 'Run the matching plotting script after computing each selected case.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_cases = sorted( CASES.keys() ) if args.case == 'all' else [ args.case ]
    for case_name in selected_cases:
        run_case( case_name, args.nmax, args.plot )


if __name__ == '__main__':
    main()
