#!/usr/bin/env python3

from fenics import *
import argparse
import numpy as np
import pickle
from pathlib import Path

class Lift1Dto2D(UserExpression):
    def __init__(self, u1D, **kwargs):
        self.u1D = u1D
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[ 0 ] = self.u1D( x[ 1 ] )  # e.g. extend along x-axis
        values[ 1 ] = 0.0

    def value_shape(self):
        return ( 2, ) 
    
class Lift1Dto2D2(UserExpression):
    def __init__(self, u1D, **kwargs):
        self.u1D = u1D
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[ 0 ] = self.u1D( 0.0 ) * x[ 0 ]

    def value_shape(self):
        return () 
    
x1     = Expression( 'x[0]', degree = 1 )
x2     = Expression( 'x[1]', degree = 1 )

re     = Expression( '2.0 + x[1]', degree = 1 )
alphae = Expression( 'cos( t )', degree = 1, t = 0 )

F   = lambda A: ( DOLFIN_EPS + inner( A, A ) ) ** ( ( re - 2.0 ) / 4.0 ) * A
sym = lambda A: 0.5 * ( A + A.T )
D = lambda u: sym( grad( u ) )

# Suppress Newton solver output
set_log_level(30)  # WARNING level
norm = lambda x : sqrt( inner( x, x ) )

def make_mean_free(function, mesh):
    volume = assemble( Constant( 1.0 ) * dx( domain = mesh ) )
    mean = assemble( function * dx( domain = mesh ) ) / volume
    function.vector()[:] = function.vector().get_local() - mean
    return mean

def check_generated_levels(data_dir, label, nmax):
    iterations_file = Path( data_dir ) / 'iterations.pkl'
    if not iterations_file.exists():
        raise FileNotFoundError(
            f'Missing {iterations_file}. Run "python3 run_section_73.py --stage {label} --nmax {nmax}" first.'
        )

    with iterations_file.open( 'rb' ) as handle:
        metadata = pickle.load( handle )
    generated_levels = len( metadata.get( 'It', [] ) )
    required_levels = nmax - 1
    if generated_levels < required_levels:
        raise RuntimeError(
            f'{iterations_file} records only {generated_levels} generated levels, '
            f'but --nmax {nmax} needs {required_levels} levels. '
            f'Run "python3 run_section_73.py --stage {label} --nmax {nmax}" to regenerate consistent data.'
        )

def append_and_print_eocs(dictionary, key, label):
    eoc_key = key + '_EOC'
    dictionary[ eoc_key ] = []
    print( label + '-EOC:' )
    for n in range( 1, len( dictionary[ key ] ) ):
        eoc = np.log( dictionary[ key ][ n ] / dictionary[ key ][ n - 1 ] ) / np.log( 0.5 )
        print( eoc )
        dictionary[ eoc_key ].append( eoc )

def error_decay_analysis( T, nmax = 9, delta = DOLFIN_EPS, alpha = Constant( 0.0 ), r = Constant( 2.0 ), u0 = Constant( 0.0 ), L = 20, R = 0.5, one_d_dir = 'Comparison1D', two_d_dir = 'Comparison', output_file = 'errors.pkl' ):

    check_generated_levels( one_d_dir, 'one-d', nmax )
    check_generated_levels( two_d_dir, 'two-d', nmax )

    dictionary = {
        'th' : [],
        'Omega_L2' : [],
        'Omega_H1' : [],
        'Omega_Gamma' : [],
        'omega_L2' : [],
        'omega_H1' : [],
        'omega_Gamma' : [],
        'L2' : [],
        'H1' : [],
        'Gamma' : [],
        'It' : [],
    }
    
    sym = lambda A: 0.5 * ( A + A.T )
    D = lambda u: sym( grad( u ) )
    r_prime = r / ( r - 1 )
    F = lambda A: ( delta + sqrt( inner( A, A ) ) ) ** ( ( r - 2.0 ) / 2.0 ) * A
    for n in range( 1, nmax ):
        print('Step:', n)
        mesh1D = IntervalMesh( 2 ** n, -R, R ); V1D = FunctionSpace( mesh1D, "P", 2 ); Q1D = FunctionSpace( mesh1D, "R", 0 )
        #mesh2D = UnitSquareMesh( 2 ** n, 2 ** n ); V2D = VectorFunctionSpace( mesh2D, "P", 2 )
        mesh2D = RectangleMesh( Point( 0.0, -R ), Point( L, R ), 2 ** n, 2 ** n ); V2D = VectorFunctionSpace( mesh2D, "P", 2 ); Q2D = FunctionSpace( mesh2D, "P", 1 )
        mesh2D_short = RectangleMesh( Point( L / 4.0, -R ), Point( 3.0 * L / 4.0, R ), max( 1, 2 ** ( n - 1 ) ), 2 ** n );
        num_steps = 2 ** n; dt =  T / num_steps  

        V2D_short = VectorFunctionSpace( mesh2D_short, "P", 2 )
        Q2D_short = FunctionSpace( mesh2D_short, "P", 1 )
        omega_volume = assemble( Constant( 1.0 ) * dx( domain = mesh2D_short ) )

        Omega_L2_err = []
        Omega_H1_err = []
        Omega_Gamma_err = []
        omega_L2_err = []
        omega_H1_err = []
        omega_Gamma_err = []
        for i in range( 1, num_steps + 1 ):
            u1D  = Function( V1D, one_d_dir + "/v" + str( n ) + "/" + str( i ) + ".xml" );     u1D  = interpolate( Lift1Dto2D( u1D, degree = 2 ), V2D )
            pi1D = Function( Q1D, one_d_dir + "/Gamma" + str( n ) + "/" + str( i ) + ".xml" ); pi1D = interpolate( Lift1Dto2D2( pi1D, degree = 1 ), Q2D )
            u2D  = Function( V2D, two_d_dir + "/v" + str( n ) + "/" + str( i ) + ".xml" )
            pi2D = Function( Q2D, two_d_dir + "/pi" + str( n ) + "/" + str( i ) + ".xml" )

            # Pressure is defined modulo constants. Fix both gauges on the full
            # strip Omega for the full-domain pressure-potential error.
            make_mean_free( pi1D, mesh2D )
            make_mean_free( pi2D, mesh2D )

            u1D_short  = interpolate( u1D, V2D_short )
            u2D_short  = interpolate( u2D, V2D_short )
            pi1D_short = interpolate( pi1D, Q2D_short )
            pi2D_short = interpolate( pi2D, Q2D_short )

            #plot( project( pi1D - pi2D, Q2D ) ); plt.show()

            #if i == 0: 
            #    #plot_obj = plot( project( dot( u1D - u2D,  u1D - u2D ), FunctionSpace( mesh2D_short, 'P', 1 ) ), figsize= (8,2) ); plt.colorbar(plot_obj); plt.show()
            #    x = np.linspace( 0, L, 100)
            #    y1 = np.array([pi1D([xi,0.0])  for xi in x]) 
            #    y2 = np.array([pi2D([xi,0.0])  for xi in x]) 
            #    _, ax = plt.subplots(figsize=(10,5))
            #    ax.plot( x,y1,c='b')
            #    ax.plot( x,y2,c='r'); plt.show()

            #print( sqrt( assemble( ( dot( u2D, Constant((0.0,1.0)) ) ) ** 2.0 * dx( domain = mesh2D_short ) ) ) )
            #u2D_short = interpolate( u2D, VectorFunctionSpace( mesh2D_short, 'P', 2 ) )
            #print( sqrt( assemble( dot( div( outer( u2D_short, u2D_short ) ), div( outer( u2D_short, u2D_short ) ) ) * dx( domain = mesh2D_short ) ) ) )

            #print( assemble( dot( u1D - u2D, u1D - u2D ) * dx( domain = mesh2D ) ) ) 
            #pi2D_short = 0.0
            #mean = assemble( pi1D * dx( domain = mesh2D ) )
            Omega_L2_err.append( assemble( dot( u1D - u2D, u1D - u2D ) * dx( domain = mesh2D ) ) )
            Omega_H1_err.append( dt * assemble( inner( F( D( u1D ) ) - F( D( u2D ) ), F( D( u1D ) ) - F( D( u2D ) ) ) * dx( domain = mesh2D ) ) )
            pi_difference_Omega = pi1D - pi2D
            pi_modular_difference_Omega = abs( pi_difference_Omega )
            Omega_Gamma_err.append( dt * assemble( ( ( delta + norm( D( u1D ) ) + norm( D( u2D ) ) ) ** ( r - 1.0 ) + pi_modular_difference_Omega ) ** ( r_prime - 2.0 )\
                                               * ( pi_modular_difference_Omega ** 2.0 ) * dx( domain = mesh2D ) ) )

            omega_L2_err.append( assemble( dot( u1D_short - u2D_short, u1D_short - u2D_short ) * dx( domain = mesh2D_short ) ) )
            omega_H1_err.append( dt * assemble( inner( F( D( u1D_short ) ) - F( D( u2D_short ) ), F( D( u1D_short ) ) - F( D( u2D_short ) ) ) * dx( domain = mesh2D_short ) ) )
            pi_difference_omega = pi1D_short - pi2D_short
            omega_mean = assemble( pi_difference_omega * dx( domain = mesh2D_short ) ) / omega_volume
            pi_difference_omega_centered = pi_difference_omega - omega_mean
            pi_modular_difference_omega = abs( pi_difference_omega_centered )
            omega_Gamma_err.append( dt * assemble( ( ( delta + norm( D( u1D_short) ) + norm( D( u2D_short ) ) ) ** ( r - 1.0 ) + pi_modular_difference_omega ) ** ( r_prime - 2.0 )\
                                               * ( pi_modular_difference_omega ** 2.0 ) * dx( domain = mesh2D_short ) ) )

        dictionary[ 'Omega_L2' ].append( sqrt( np.amax( np.array( Omega_L2_err ) ) ) )
        dictionary[ 'Omega_H1' ].append( sqrt( np.sum( Omega_H1_err ) ) )
        dictionary[ 'Omega_Gamma' ].append( sqrt( np.sum( Omega_Gamma_err ) ) )
        dictionary[ 'omega_L2' ].append( sqrt( np.amax( np.array( omega_L2_err ) ) ) )
        dictionary[ 'omega_H1' ].append( sqrt( np.sum( omega_H1_err ) ) )
        dictionary[ 'omega_Gamma' ].append( sqrt( np.sum( omega_Gamma_err ) ) )

        dictionary[ 'L2' ].append( dictionary[ 'omega_L2' ][ -1 ] )
        dictionary[ 'H1' ].append( dictionary[ 'omega_H1' ][ -1 ] )
        dictionary[ 'Gamma' ].append( dictionary[ 'omega_Gamma' ][ -1 ] )
        dictionary[ 'th' ].append( dt + mesh2D.hmax() ) 

    append_and_print_eocs( dictionary, 'Omega_L2', 'Omega_L2' )
    append_and_print_eocs( dictionary, 'Omega_H1', 'Omega_H1' )
    append_and_print_eocs( dictionary, 'Omega_Gamma', 'Omega_Gamma' )
    append_and_print_eocs( dictionary, 'omega_L2', 'omega_L2' )
    append_and_print_eocs( dictionary, 'omega_H1', 'omega_H1' )
    append_and_print_eocs( dictionary, 'omega_Gamma', 'omega_Gamma' )

    dictionary[ 'L2_EOC' ] = list( dictionary[ 'omega_L2_EOC' ] )
    dictionary[ 'H1_EOC' ] = list( dictionary[ 'omega_H1_EOC' ] )
    dictionary[ 'Gamma_EOC' ] = list( dictionary[ 'omega_Gamma_EOC' ] )


    with open( output_file, 'wb' ) as f:
        pickle.dump(dictionary, f)
 

def parse_args():
    parser = argparse.ArgumentParser( description = 'Compare the reduced 1D and full 2D Section 7.3 approximations.' )
    parser.add_argument( '--nmax', type = int, default = 9, help = 'Upper refinement loop bound. Runs n=1,...,nmax-1; paper setup: nmax=9.' )
    parser.add_argument( '--one-d-dir', default = 'Comparison1D', help = 'Directory with reduced 1D snapshots.' )
    parser.add_argument( '--two-d-dir', default = 'Comparison', help = 'Directory with full 2D snapshots.' )
    parser.add_argument( '--output-file', default = 'errors.pkl', help = 'Pickle file for comparison errors.' )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    error_decay_analysis( 2.0 * np.pi, nmax = args.nmax, delta = DOLFIN_EPS, alpha = alphae, r = re, u0 = Constant( ( 0.0, 0.0 ) ), L = 20, R = 0.5, one_d_dir = args.one_d_dir, two_d_dir = args.two_d_dir, output_file = args.output_file )
