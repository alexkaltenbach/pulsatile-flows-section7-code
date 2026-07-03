from fenics import *
import numpy as np
import scipy.integrate as integrate
 
class Re(UserExpression):
    def __init__(self, ps, ts, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ps = ps 
        self.ts = ts

    def eval(self, values, x) -> None:
        for i in range(1, len(self.ts)):
            if abs(x[0]) >= abs(self.ts[i]) and abs(x[0]) < abs(self.ts[i - 1]):
                values[0] = self.ps[i - 1]
            if near(abs(x[0]), abs(self.ts[0])):
                values[0] = self.ps[0]

    def value_shape(self) -> tuple:
        return ()
 
class Ue(UserExpression):
    def __init__(self, ps, ts, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ps = ps 
        self.ts = ts
        self.ps_prime = [self.ps[i] / (self.ps[i] - 1.0) for i in range(len(self.ps))]
        self.cs = [-1.0 / self.ps_prime[0] * (-self.ts[0]) ** self.ps_prime[0]]
        for i in range(1, len(self.ps)):
            c_prev = self.cs[-1]
            self.cs.append(1.0 / self.ps_prime[i - 1] * ((-self.ts[i]) ** self.ps_prime[i - 1]) + c_prev - 1.0 / self.ps_prime[i] * ((-self.ts[i]) ** self.ps_prime[i]))

    def eval(self, values, x) -> None:
        for i in range(1, len(self.ts)):
            if abs(x[0]) >= abs(self.ts[i]) and abs(x[0]) < abs(self.ts[i - 1]):
                values[0] = -(1.0 / self.ps_prime[i - 1] * (abs(x[0]) ** self.ps_prime[i - 1]) + self.cs[i - 1]) 
            if near(abs(x[0]), 1.0):
                values[0] = 0.0 

    def value_shape(self) -> tuple:
        return ()
    
class DUe(UserExpression):
    def __init__(self, ps, ts, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ps = ps 
        self.ts = ts
        self.ps_prime = [self.ps[i] / (self.ps[i] - 1.0) for i in range(len(self.ps))]
        self.cs = [-1.0 / self.ps_prime[0] * (-self.ts[0]) ** self.ps_prime[0]]
        for i in range(1, len(self.ps)):
            c_prev = self.cs[-1]
            self.cs.append(1.0 / self.ps_prime[i - 1] * ((-self.ts[i]) ** self.ps_prime[i - 1]) + c_prev - 1.0 / self.ps_prime[i] * ((-self.ts[i]) ** self.ps_prime[i]))

    def eval(self, values, x) -> None:
        for i in range(1, len(self.ts)):
            if abs(x[0]) > DOLFIN_EPS:
                if abs(x[0]) >= abs(self.ts[i]) and abs(x[0]) < abs(self.ts[i - 1]):
                    values[0] = -abs(x[0]) ** ( self.ps_prime[i - 1] - 2.0 ) * x[0]
                if near(abs(x[0]), 1.0):
                    values[0] = 0.0 
            else:
                values[0] = 0.0

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
    def __init__(self, ps, ts, **kwargs):
        super().__init__(**kwargs)
        self.ps = ps 
        self.ts = ts
        self.ps_prime = [self.ps[i] / (self.ps[i] - 1.0) for i in range(len(self.ps))]
        self.cs = [-1.0 / self.ps_prime[0] * (-self.ts[0]) ** self.ps_prime[0]]
        for i in range(1, len(self.ps)):
            c_prev = self.cs[-1]
            self.cs.append(1.0 / self.ps_prime[i - 1] * ((-self.ts[i]) ** self.ps_prime[i - 1]) + c_prev - 1.0 / self.ps_prime[i] * ((-self.ts[i]) ** self.ps_prime[i]))

    def eval(self, values, x): 
        def f( x ):
            for i in range(1, len(self.ts)):
                if abs(x) >= abs(self.ts[i]) and abs(x) < abs(self.ts[i - 1]):
                    return -(1.0 / self.ps_prime[i - 1] * (abs(x) ** self.ps_prime[i - 1]) + self.cs[i - 1]) 
                if near(abs(x), 1.0):
                    return 0.0 
         
        values[0] = integrate.quad(f, -1.0, 1.0)[0]

    def value_shape(self):
        return ()
