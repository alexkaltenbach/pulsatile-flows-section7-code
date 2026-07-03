#!/usr/bin/env python3

from fenics import *
from PicardIteration import PicardIteration
import argparse
import numpy as np
import os
import pickle

re     = Expression( '2.0 + x[1]', degree = 1 )
alphae = Expression( 'cos( t )', degree = 1, t = 0 )

# Suppress Newton solver output
set_log_level(30)  # WARNING level
L = 20
R = 0.5

def run_error_decay_experiment( T, nmax = 9, delta = DOLFIN_EPS, alpha = Constant( 0.0 ), r = Constant( 2.0 ), u0 = Constant( 0.0 ), L = 20, R = 0.5, output_dir = 'Comparison' ):

    dictionary = { 'It' : [], 'nmax' : nmax, 'levels' : list( range( 1, nmax ) ) }
    os.makedirs( output_dir, exist_ok = True )
     
    for n in range( 1, nmax ):
        print('Step:', n)
        mesh      = RectangleMesh( Point( 0.0, -R ), Point( L, R ), 2 ** n, 2 ** n )
        num_steps = 2 ** n; dt =  T / num_steps 
        snap_list, _, _, num_iter = PicardIteration( mesh, T, num_steps, alpha = alpha, delta = delta, r = r, u0 = u0, snapshots = True, info = True,  L = L, R = R )

        os.makedirs( output_dir + "/v" + str( n ), exist_ok = True )
        os.makedirs( output_dir + "/pi" + str( n ), exist_ok = True )
        for i in range( len( snap_list ) ):
            File( output_dir + "/v" + str( n ) + "/" + str( i ) + ".xml" ) << snap_list[ i ][ 0 ]
            File( output_dir + "/pi" + str( n ) + "/" + str( i ) + ".xml" ) << snap_list[ i ][ 1 ]
        dictionary['It'].append( num_iter )

    with open( output_dir + '/iterations.pkl', 'wb' ) as f:
        pickle.dump(dictionary, f)

def parse_args():
    parser = argparse.ArgumentParser( description = 'Run the full 2D Section 7.3 experiment.' )
    parser.add_argument( '--nmax', type = int, default = 9, help = 'Upper refinement loop bound. Runs n=1,...,nmax-1; paper setup: nmax=9.' )
    parser.add_argument( '--output-dir', default = 'Comparison', help = 'Directory for 2D snapshots.' )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_error_decay_experiment( 2.0 * np.pi, nmax = args.nmax, delta = DOLFIN_EPS, alpha = alphae, r = re, u0 = Constant( ( 0.0, 0.0 ) ), L = L, R = R, output_dir = args.output_dir )
