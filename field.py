import numpy as np
from abc import ABC, abstractmethod

class Field(ABC):
    @abstractmethod
    def check_member(self, a):
        pass

    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def mult(self, a, b):
        pass

    @abstractmethod
    def inv_mult(self, a, b):
        pass

class PrimeField(Field):
    ''' Prime finite field (F2, F3, ...)
    Attributes: Elements & Prime values
    Method: Add, Multiply and Inv_Mult (Division)
    
    Subtraction is left out, as the operations is the
    same as addition for prime finite field. 
    '''
    def __init__(self, prime: int):
        self.elements = np.arange(prime)
        self.prime = prime
        self.inv = {}
        for elem in self.elements:
            if elem != 0 and elem not in self.inv:
                for inv in self.elements:
                    if inv != 0 and inv not in self.inv \
                        and self.mult(inv, elem) == 1:
                        self.inv[elem] = inv
                        self.inv[inv] = elem
                        break
    
    def check_member(self, a):
        return a in self.elements
    
    def add(self, a, b):
        return((a + b) % self.prime)
    
    def mult(self, a, b):
        return((a * b) % self.prime)
    
    def inv_mult(self, a, b):
        return((a * self.inv[b]) % self.prime)
        
class RealField(Field):
    
    def __init__(self):
        pass

    def check_member(self, a):
        return type(a) is int or type(a) is float

    def add(self, a, b):
        return a + b
    
    def mult(self, a, b):
        return a * b
    
    def inv_mult(self, a, b):
        return a / b

class ComplexField(RealField):
    ''' Same operators as Real'''
    def check_member(self, a):
        return type(a) is complex

# Testing
# real = RealField()
# print(real.mult(0.5, np.array([0, 2, 0, -1])))