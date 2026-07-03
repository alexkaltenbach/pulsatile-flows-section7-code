#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_command(args, cwd, dry_run=False):
    print(f'[{cwd.name}] ' + ' '.join(args))
    if not dry_run:
        subprocess.run(args, cwd=str(cwd), check=True)


def section_71_args(args):
    nmax = 6 if args.quick else args.nmax71
    command = [
        sys.executable,
        'run_section_71.py',
        '--radius',
        'all',
        '--nmax',
        str(nmax),
    ]
    if args.plots:
        command.append('--plot')
    return command


def section_72_args(args):
    nmax = 6 if args.quick else args.nmax72
    command = [
        sys.executable,
        'run_section_72.py',
        '--case',
        'all',
        '--nmax',
        str(nmax),
    ]
    if args.plots:
        command.append('--plot')
    return command


def section_73_args(args):
    nmax = 5 if args.quick else args.nmax73
    command = [
        sys.executable,
        'run_section_73.py',
        '--stage',
        'all',
        '--nmax',
        str(nmax),
    ]
    if not args.plots:
        command.append('--skip-plots')
    return command


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the numerical experiments from Sections 7.1-7.3.'
    )
    parser.add_argument(
        '--section',
        choices=['all', '71', '72', '73'],
        default='all',
        help='Select which section to run.',
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Open the corresponding plots after computations.',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Explicitly suppress plot windows. This is the default.',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use smaller refinement levels for a smoke test.',
    )
    parser.add_argument('--nmax71', type=int, default=12)
    parser.add_argument('--nmax72', type=int, default=10)
    parser.add_argument('--nmax73', type=int, default=9)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plots and args.no_plots:
        raise ValueError('Use either --plots or --no-plots, not both.')

    selected_sections = ['71', '72', '73'] if args.section == 'all' else [args.section]
    commands = {
        '71': (ROOT / 'pLaplaceMixed', section_71_args(args)),
        '72': (ROOT / 'pLaplaceMixed', section_72_args(args)),
        '73': (ROOT / 'pNavierStokesMixed', section_73_args(args)),
    }

    for section in selected_sections:
        cwd, command = commands[section]
        if not cwd.exists():
            raise FileNotFoundError(cwd)
        run_command(command, cwd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
