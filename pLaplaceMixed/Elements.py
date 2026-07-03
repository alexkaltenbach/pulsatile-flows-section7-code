from fenics import *
from ufl import interval

def StableElement( mesh, l = 1 ):
    V_element = FiniteElement( 'P', interval, l )
    R_element = FiniteElement( 'R', interval, 0 )
    W_element = MixedElement( [ V_element, R_element ] )
    W = FunctionSpace( mesh, W_element )
    return W