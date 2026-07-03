from fenics import *
import numpy as np
from Operators import * 
from Elements import *  

def solve_plaplace( mesh, T, num_steps,
                   l     = 1,
                   alpha = Constant( 0.0 ), 
                   uD    = Constant( 0.0 ),
                   delta = DOLFIN_EPS, 
                   r     = Constant( 2.0 ), 
                   stress_factor = Constant( 1.0 ),
                   u0    = Constant( 0.0 ), snapshots = False, track_point = None,
                   info  = False, inner_info = False, nonlinear_tol = 1e-8,
                   nonlinear_max_iter = 50 ):
    
    dt = T / num_steps
    W  = StableElement( mesh, l = l )
    V  = FunctionSpace( mesh, 'P', l )
    
    bc = DirichletBC( W.sub(0), uD, 'on_boundary') 

    w_sol = Function( W )
    u_n = Function( V ); u_n.assign( interpolate( u0, V ) )

    ( u, Gamma ) = TrialFunctions( W )
    ( v, eta )   = TestFunctions( W )

    snap_list = []   
    time_series = []   

    if snapshots:
        snap_list   = [ [ u_n.copy( deepcopy = True ), None ] ]
    if track_point is not None:
        time_series = [ u_n( track_point ) ]

    vol_Sigma = assemble( Constant( 1.0 ) * dx( domain = mesh ) )

    for n in range( 1, num_steps + 1 ):
        t = n * dt 
        if hasattr( alpha, 't' ):
            alpha.t = t
        if hasattr( uD, 't' ):
            uD.t = t

        a  = (1.0/dt) * u * v * dx( domain = mesh )\
            + dot( grad( u ), grad( v ) ) * dx( domain = mesh )\
            + FluxConstraint( u, Gamma, v, eta, mesh ) 

        L  = (1.0/dt) * u_n * v * dx( domain = mesh )\
             + alpha / vol_Sigma * eta * dx( domain = mesh )
        
        solve(a == L, w_sol, bc)
 
        w_prev   = Function( W )
        w_prev.assign( w_sol )
        u_prev, _ = split( w_prev )
        u_sol, Gamma_sol = w_sol.split( deepcopy = True )
        A = lambda a, b: stress_factor * ( delta + sqrt( dot( b, b ) ) ) ** ( r - 2.0 ) * a
        err = 1.0
        inner_iter = 0
        if not inner_info:
            set_log_active( False )
        while( err > nonlinear_tol and inner_iter < nonlinear_max_iter ):
            a = ( 1.0 + (1.0/dt) ) * u * v * dx( domain = mesh ) \
                + dot( A( grad( u ), grad( u_prev ) ), grad( v ) ) * dx( domain = mesh )\
                + FluxConstraint( u, Gamma, v, eta, mesh ) 
            
            L = alpha / vol_Sigma * eta * dx( domain = mesh )\
                + u_prev * v * dx( domain = mesh )\
                + (1.0/dt) * u_n * v * dx( domain = mesh )
            
            solve( a == L, w_sol, bc )
            u_sol, Gamma_sol = w_sol.split( deepcopy = True )
            err = np.sqrt( assemble( ( u_sol - u_prev ) ** 2 * dx( domain = mesh ) ) )
            if inner_info:
                print( f'      nonlinear iter {inner_iter + 1:2d}: residual = {err:.3e}' )
            w_prev.assign( w_sol ) 
            inner_iter += 1
        if inner_info:
            print( f'      nonlinear iterations = {inner_iter:2d}, residual = {err:.3e}' )
        elif info:
            print( f'Time step {n:4d}/{num_steps}: nonlinear iterations = {inner_iter:2d}, residual = {err:.3e}' )
        if inner_iter == nonlinear_max_iter and err > nonlinear_tol:
            print( f'Warning: nonlinear iteration stopped at {err:.3e} after {nonlinear_max_iter} iterations.' )
        u_sol, Gamma_sol = w_sol.split( deepcopy = True ) 

        u_n.vector()[:] = u_sol.vector().get_local() 

        if snapshots:
            snap_list.append( [ u_sol.copy( deepcopy = True ), Gamma_sol.copy( deepcopy = True ) ] )
        if track_point is not None:
            time_series.append( u_n( track_point ) )

    return u_n, snap_list, time_series
