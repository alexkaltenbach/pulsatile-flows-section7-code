from fenics import * 

def FluxConstraint( uh, Gammah, vh, etah, mesh ):
    return   uh     * etah * dx( domain = mesh )\
           + Gammah * vh   * dx( domain = mesh ) 
