from fenics import * 

def IncompressibilityConstraint( uh, ph, vh, qh, mesh ):
    return   div( uh ) * qh * dx( domain = mesh )\
           - div( vh ) * ph * dx( domain = mesh ) 

def MeanConstraint( ph, lmbdah, qh, muh, mesh ):
    return    ph * muh * dx( domain = mesh )\
            + qh * lmbdah * dx( domain = mesh ) 

def FluxConstraint1( uh, gammah1, vh, nuh1, dS_line ):
    return gammah1 * dot( vh, Constant( ( 1.0, 0.0 ) ) ) * dS_line( 1 )\
            + nuh1 * dot( uh, Constant( ( 1.0, 0.0 ) ) ) * dS_line( 1 )

def FluxConstraint2( uh, gammah2, vh, nuh2, dS_line ):
    return gammah2 * dot( vh, Constant( ( 1.0, 0.0 ) ) ) * dS_line( 2 )\
            + nuh2 * dot( uh, Constant( ( 1.0, 0.0 ) ) ) * dS_line( 2 )
