from fenics import *
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import newton
  
class Re(UserExpression):
    def __init__(self, p1, p2, R, xi, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p1 = p1
        self.p2 = p2
        self.R = R
        self.xi = xi

    def eval(self, values, x) -> None:
        if x[0] <= self.xi:
            values[0] = self.p1
        else:
            values[0] = self.p2

    def value_shape(self) -> tuple:
        return ()
 
class Ue(UserExpression):
    def __init__(self, p1, p2, R, xi, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p1 = p1
        self.p1_prime = p1 / (p1 - 1.0)
        self.p2 = p2
        self.p2_prime = p2 / (p2 - 1.0)
        self.R = R
        self.xi = xi

        def func(b):
            xi = float(self.xi)
            R = float(self.R)
            p1p = float(self.p1_prime)
            p2p = float(self.p2_prime)
            b = float(b)

            return (-1.0 / p1p * abs(xi - b) ** p1p
                    + 1.0 / p1p * abs(R + b) ** p1p
                    + 1.0 / p2p * abs(xi - b) ** p2p
                    - 1.0 / p2p * abs(R - b) ** p2p)

        self.b = newton(func, 0.0)
        #print( self.b )
        self.c1 = 1.0 / self.p1_prime * (np.abs(self.R + self.b) ** self.p1_prime)
        self.c2 = 1.0 / self.p2_prime * (np.abs(self.R - self.b) ** self.p2_prime)

    def eval(self, values, x) -> None:
        if x[0] <= self.xi:
            values[0] = -1.0 / self.p1_prime * (abs(x[0] - self.b) ** self.p1_prime) + self.c1 
        else:
            values[0] = -1.0 / self.p2_prime * (abs(x[0] - self.b) ** self.p2_prime) + self.c2 

    def value_shape(self) -> tuple:
        return ()
    
class DUe(UserExpression):
    def __init__(self, p1, p2, R, xi, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p1 = p1
        self.p1_prime = p1 / (p1 - 1.0)
        self.p2 = p2
        self.p2_prime = p2 / (p2 - 1.0)
        self.R = R
        self.xi = xi

        def func(b):
            xi = float(self.xi)
            R = float(self.R)
            p1p = float(self.p1_prime)
            p2p = float(self.p2_prime)
            b = float(b)

            return ( -1.0 / p1p * abs(xi - b) ** p1p
                    + 1.0 / p1p * abs(R + b) ** p1p
                    + 1.0 / p2p * abs(xi - b) ** p2p
                    - 1.0 / p2p * abs(R - b) ** p2p)

        self.b = newton(func, 0.0)
        self.c1 = 1.0 / self.p1_prime * (np.abs(self.R + self.b) ** self.p1_prime)
        self.c2 = 1.0 / self.p2_prime * (np.abs(self.R - self.b) ** self.p2_prime)

    def eval(self, values, x) -> None:
        if abs(x[0] - self.b) <= DOLFIN_EPS:
            values[0] = 0.0
        elif x[0] <= self.xi:
            values[0] = -(abs(x[0] - self.b) ** ( self.p1_prime - 2.0 )) * (x[0] - self.b)
        else:
            values[0] = -(abs(x[0] - self.b) ** ( self.p2_prime - 2.0 )) * (x[0] - self.b)
        
    def value_shape(self) -> tuple:
        return (1,)
    
class Gammae(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

    def eval(self, values, x): 
        values[0] = -1.0

    def value_shape(self):
        return ()
    
class Alphae(UserExpression):
    def __init__(self, p1, p2, R, xi, **kwargs):
        super().__init__(**kwargs) 
        self.p1 = p1
        self.p1_prime = p1 / (p1 - 1.0)
        self.p2 = p2
        self.p2_prime = p2 / (p2 - 1.0)
        self.R = R
        self.xi = xi

        def func(b):
            xi = float(self.xi)
            R = float(self.R)
            p1p = float(self.p1_prime)
            p2p = float(self.p2_prime)
            b = float(b)

            return (-1.0 / p1p * abs(xi - b) ** p1p
                    + 1.0 / p1p * abs(R + b) ** p1p
                    + 1.0 / p2p * abs(xi - b) ** p2p
                    - 1.0 / p2p * abs(R - b) ** p2p)

        self.b = newton(func, 0.0)
        self.c1 = 1.0 / self.p1_prime * (np.abs(self.R + self.b) ** self.p1_prime)
        self.c2 = 1.0 / self.p2_prime * (np.abs(self.R - self.b) ** self.p2_prime)

    def eval(self, values, x): 
        def f( x ):
            if x <= self.xi:
                return -1.0 / self.p1_prime * (abs(x - self.b) ** self.p1_prime) + self.c1 
            else:
                return -1.0 / self.p2_prime * (abs(x - self.b) ** self.p2_prime) + self.c2 
         
        values[0] = integrate.quad(f, -1.0, 1.0)[0]

    def value_shape(self):
        return ()
