from fenics import * 
#from Stationary import *
from DataNonSym import * 
from PicardIteration import *
import numpy as np  
import os
import pickle 

# Suppress Newton solver output
set_log_level(30)  # WARNING level

p1, p2 = 2.5, 1.5
R,  xi = 1.0, 0.5

filename = 'NonSym2'
 

def error_decay_analysis( T, nmax = 5, l = 1, delta = DOLFIN_EPS, r = None, u0 = Constant( 0.0 ), case_p1 = p1, case_p2 = p2, case_R = R, case_xi = xi, output_dir = filename ):
    os.makedirs( output_dir, exist_ok = True )
    r = Re( case_p1, case_p2, case_R, case_xi, degree = 6 ) if r is None else r
    alpha_value = Alphae( case_p1, case_p2, case_R, case_xi )( 0.0 )

    dictionary = { 'th' : [], 'L2' : [], 'H1' : [], 'Gamma' : [], 'It' : [], 'L2_EOC' : [], 'H1_EOC' : [], 'Gamma_EOC' : [], 'Gamma_Modular_EOC' : [] }
     
    for n in range( 1, nmax ):
        print('Step:', n)
        mesh      = IntervalMesh( 2 ** n, -1.0, 1.0 )
        num_steps = 2 ** n; dt =  T / num_steps 
        alphae     = Constant( alpha_value )
        snap_list, _, _, num_iter = PicardIteration( mesh, T, num_steps, l = l, alpha = alphae, delta = delta, r = r, u0 = u0, snapshots = True, info = True, inner_info = False )


        L2_err = []
        H1_err = []
        Gamma_err = []
        r_prime = r / ( r - 1 )
        F = lambda A: ( delta + sqrt( dot( A, A ) ) ) ** ( ( r - 2.0 ) / 2.0 ) * A
        ue     = Ue( case_p1, case_p2, case_R, case_xi, degree = 6 )
        gammae = Gammae( degree = 6 )
        Due = DUe( case_p1, case_p2, case_R, case_xi, degree = 6 )
        T_quad_element = VectorElement( family = 'Quadrature', cell = mesh.ufl_cell(), degree = 6, quad_scheme = 'default' )
        T_quad_space   = FunctionSpace( mesh, T_quad_element )
        Due_quad       = interpolate( Due, T_quad_space )

        for vh, _ in snap_list:
            L2_err.append( assemble( ( ue - vh ) * ( ue - vh ) * dx( domain = mesh ) ) )

        for vh, gammah in snap_list[ 1: ]:
            gamma_diff = abs( gammae - gammah )
            H1_err.append( dt * assemble( dot( F( Due_quad ) - F( grad( vh ) ), F( Due_quad ) - F( grad( vh ) ) ) * dx( domain = mesh, scheme = 'default', degree = 6 ) ) )
            Gamma_err.append( dt * assemble( ( ( delta + sqrt( dot( Due_quad, Due_quad ) ) ) ** ( r - 1.0 ) + gamma_diff ) ** ( r_prime - 2.0 )\
                                           * ( gamma_diff ** 2.0 ) * dx( domain = mesh, scheme = 'default', degree = 6 ) ) )

        dictionary[ 'L2' ].append( sqrt( np.amax( np.array( L2_err ) ) ) )
        dictionary[ 'H1' ].append( sqrt( np.sum( H1_err ) ) )
        dictionary[ 'Gamma' ].append( np.sum( Gamma_err ) )
        dictionary[ 'It' ].append( num_iter )
        dictionary[ 'th' ].append( dt + mesh.hmax() )
        print( 'Picard-Iterations:', num_iter )

        File("./" + output_dir  + "/v" + str( n ) + ".xml") << snap_list[ - 1 ][ 0 ]
        File("./" + output_dir  + "/Gamma" + str( n ) + ".xml") << snap_list[ - 1 ][ 1 ]

    print( 'L2-EOC:' )
    for n in  range( 1, nmax - 1 ):
        L2_eoc = np.log( dictionary[ 'L2' ][ n ] / dictionary[ 'L2' ][ n - 1 ] ) / np.log( 0.5 )
        print( L2_eoc )
        dictionary[ 'L2_EOC' ].append( L2_eoc )
    
    print( 'H1-EOC:' )
    for n in  range( 1, nmax - 1 ):
        H1_eoc = np.log( dictionary[ 'H1' ][ n ] / dictionary[ 'H1' ][ n - 1 ] ) / np.log( 0.5 )
        print( H1_eoc )
        dictionary[ 'H1_EOC' ].append( H1_eoc )

    print( 'Gamma^{1/2}-EOC:' )
    for n in  range( 1, nmax - 1 ):
        Gamma_eoc = np.log( sqrt( dictionary[ 'Gamma' ][ n ] ) / sqrt( dictionary[ 'Gamma' ][ n - 1 ] ) ) / np.log( 0.5 )
        Gamma_modular_eoc = np.log( dictionary[ 'Gamma' ][ n ] / dictionary[ 'Gamma' ][ n - 1 ] ) / np.log( 0.5 )
        print( Gamma_eoc )
        dictionary[ 'Gamma_EOC' ].append( Gamma_eoc )
        dictionary[ 'Gamma_Modular_EOC' ].append( Gamma_modular_eoc )

    with open("./" + output_dir  + "/errors.pkl", 'wb') as f:
        pickle.dump(dictionary, f)
