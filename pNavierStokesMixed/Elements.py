from fenics import *

def StableElement( mesh ): 
    V_element = VectorElement( "P", triangle, 2 )
    Q_element = FiniteElement( "P", triangle, 1 )
    R_element = FiniteElement( "R", triangle, 0 )
    R_element = FiniteElement( "R", triangle, 0 )
    R_element = FiniteElement( "R", triangle, 0 )
    W_element = MixedElement( [ V_element, Q_element, R_element, R_element, R_element ] )
    W         = FunctionSpace( mesh, W_element )
    return W