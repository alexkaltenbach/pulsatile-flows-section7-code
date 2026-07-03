from fenics import *
import numpy as np
from Operators import * 
from Elements import *  

def solve_pnavier_stokes( mesh, T, num_steps, 
                         alpha = Constant( 0.0 ), 
                         uD    = Constant( 0.0 ),
                         delta = DOLFIN_EPS, 
                         r     = Constant( 2.0 ), 
                         u0    = Constant( ( 0.0, 0.0 ) ), snapshots = False, track_point = None,  L = 10, R = 0.5,
                         info = False, inner_info = False, nonlinear_tol = 1e-8,
                         nonlinear_max_iter = 50 ):
    dt = T / num_steps
    W  = StableElement( mesh )
    V  = VectorFunctionSpace( mesh, 'P', 2 )
    Q  = FunctionSpace( mesh, 'P', 1 )
    
    def boundary(x, on_boundary):
        return near( x[1], -R ) or near( x[1], R )
    bc = DirichletBC( W.sub(0), Constant( ( 0.0, 0.0 ) ), boundary )

    boundaries = MeshFunction( 'size_t', mesh, mesh.topology().dim() - 1 ); boundaries.set_all( 0 )
    AutoSubDomain( lambda x: near( x[0], 0.0 ) ).mark( boundaries, 1 )
    AutoSubDomain( lambda x: near( x[0], float( L ) ) ).mark( boundaries, 2 )

    dS_line = Measure( 'ds', domain = mesh, subdomain_data = boundaries )

    Sigma_vol = assemble( Constant( 1.0 ) * dS_line( 1 ) )

    w_sol = Function( W )
    u_n   = Function( V ); u_n.assign(u0)
    p_n   = Function( Q ) 

    w = TrialFunction( W ); ( u, p, lmbda, gamma1, gamma2 ) = split(w)
    z = TestFunction( W );  ( v, q,    mu,    nu1,    nu2 ) = split(z)
        
    snap_list = []
    time_series = []

    if snapshots:
        snap_list   = [ [ u_n, p_n ] ]
    if track_point is not None:
        time_series = [ u_n( track_point ) ]

    for n in range( 1, num_steps + 1):
        t = n * dt 
        if hasattr( alpha, 't' ):
            alpha.t = t
        if hasattr( uD, 't' ):
            uD.t = t 


        # Solve Stokes
        D = lambda v : 0.5 * ( grad( v ) + grad( v ).T ) 
        a  = (1/dt) * dot( u, v ) * dx( domain = mesh )
        a += inner( D( u ), D( v ) ) * dx( domain = mesh )
        a += IncompressibilityConstraint( u, p, v, q, mesh )  
        a += MeanConstraint( p, lmbda, q, mu, mesh )
        a += FluxConstraint1( u, gamma1, v, nu1, dS_line ) 
        a += FluxConstraint2( u, gamma2, v, nu2, dS_line ) 

        L_  = (1/dt) * dot( u_n, v ) * dx( domain = mesh ) 
        L_ += alpha / Sigma_vol * nu1 * dS_line( 1 ) 
        L_ += alpha / Sigma_vol * nu2 * dS_line( 2 ) 
 
        solve(a == L_, w_sol, bc, solver_parameters={"linear_solver": "mumps"})

        w_prev = Function( W )
        w_prev.assign( w_sol )
        u_prev, _, _, _, _ = split( w_prev )
        u_sol, _, _, _, _  = w_sol.split( deepcopy = True )
        S_ = lambda A, B: ( delta + sqrt( inner( B, B ) ) ) ** ( r - 2.0 ) * A
        err = 1.0
        inner_iter = 0
        if inner_info == False:
            set_log_active( False )
        while( err > nonlinear_tol and inner_iter < nonlinear_max_iter ):  
            a = ( 1.0 + (1.0/dt) ) * dot( u, v ) * dx( domain = mesh ) \
                + inner( S_( D( u ), D( u_prev ) ), D( v ) ) * dx( domain = mesh )\
                - 0.5 * inner( outer( u_prev, u ), grad( v ) ) * dx( domain = mesh )\
                + 0.5 * inner( outer( v, u ), grad( u_prev ) ) * dx( domain = mesh )\
                + IncompressibilityConstraint( u, p, v, q, mesh )\
                + MeanConstraint( p, lmbda, q, mu, mesh )\
                + FluxConstraint1( u, gamma1, v, nu1, dS_line )\
                + FluxConstraint2( u, gamma2, v, nu2, dS_line )

            L = alpha / Sigma_vol * nu1 * dS_line( 1 )\
                + alpha / Sigma_vol * nu2 * dS_line( 2 )\
                + dot( u_prev, v ) * dx( domain = mesh )\
                + (1.0/dt) * dot( u_n, v ) * dx( domain = mesh )

            solve( a == L, w_sol, bc, solver_parameters={"linear_solver": "mumps"} )
            u_sol, p_sol, _, _, _ = w_sol.split( deepcopy = True )
            err = np.sqrt( assemble( dot( u_sol - u_prev, u_sol - u_prev ) * dx( domain = mesh ) ) )
            if inner_info:
                print( f'      nonlinear iter {inner_iter + 1:2d}: residual = {err:.3e}' )
            w_prev.assign( w_sol ) 
            inner_iter += 1
        if inner_info:
            print( f'      nonlinear iterations = {inner_iter:2d}, residual = {err:.3e}' )
        if inner_iter == nonlinear_max_iter and err > nonlinear_tol:
            print( f'Warning: nonlinear iteration stopped at {err:.3e} after {nonlinear_max_iter} iterations.' )

        u_sol, p_sol, _, _, _  = w_sol.split(deepcopy=True)
        u_n.vector()[:] = u_sol.vector().get_local()
        p_n.vector()[:] = p_sol.vector().get_local()

        if snapshots:
            snap_list.append( [ u_n.copy( deepcopy = True ), p_n.copy( deepcopy = True ) ] )
        if track_point is not None:
            time_series.append( u_n( track_point ) )

    return u_n, snap_list, time_series
