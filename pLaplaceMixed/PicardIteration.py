from fenics import *
from pxLaplaceFEM import *
import numpy as np

def PicardIteration( mesh, T, num_steps, 
                    l     = 1,
                    alpha = Constant( 0.0 ), 
                    uD    = Constant( 0.0 ), 
                    delta = DOLFIN_EPS, 
                    r     = Constant( 2.0 ), 
                    stress_factor = Constant( 1.0 ),
                    u0    = Constant( 0.0 ), snapshots =  False, info = False,
                    inner_info = False, tol = 1e-12, max_iter = 100 ):
    
    V        = FunctionSpace( mesh, 'P', l )
    u0       = interpolate( u0, V )
 
    periodicity_errors_linfty          = []
    periodicity_errors_L2              = [] 

    num_iter = 0

    for k in range(max_iter):
        num_iter+=1
        uT, snap_list, _ = solve_plaplace( mesh, T, num_steps, l = l, alpha = alpha, uD = uD, delta = delta, r = r, stress_factor = stress_factor, u0 = u0, snapshots = snapshots, track_point = None, info = False, inner_info = inner_info )
 
        periodicity_err_linfty = np.max( np.abs( uT.vector().get_local() - u0.vector().get_local() ) ); periodicity_errors_linfty.append( periodicity_err_linfty )
        periodicity_err_L2     = errornorm( uT, u0, norm_type = 'L2' );                periodicity_errors_L2.append( periodicity_err_L2 ) 
        
        if info:
            print( fr'Iteration {k:2d}: periodicity error (linfty) = {periodicity_err_linfty:.3e}, periodicity error (L2) = {periodicity_err_L2:.3e}' )
        u0.assign( uT )
        if periodicity_err_L2 < tol:
            if info:
                print( 'Converged.' )
            break
    else:
        print( f'Warning: Picard iteration stopped after {max_iter} iterations with periodicity error {periodicity_errors_L2[-1]:.3e}.' )

    if snapshots:
        return snap_list, periodicity_errors_linfty, periodicity_errors_L2, num_iter 
    else:
        return u0, periodicity_errors_linfty, periodicity_errors_L2, num_iter
