import numpy as np
from abc import abstractmethod, ABC
from vectorspace import *
from field import *

class LinearTransform(ABC):
    ''' A linear transformation T: V -> W with
    V and W sharing the same F is any T that:
    1. T(a) + T(b) = T(a+b)
    2. k T(a) = T(ka) (k: scalar; a, b: vectors)

    Can't check for the 2 properties :D 
    This class only defines the basic of a transformation
    '''
    def __init__(self, domain: VectorSpace, co_domain: VectorSpace, field: Field):
        self.field = field
        self.domain = domain
        self.co_domain = co_domain
    
    @abstractmethod
    def func(self, a):
        ''' Defining the image in W of any vector a in V.
        For row/sol/col space as domain, a: n x 1 vector. 
        Take transpose if needed. Return a column np.array vector
        for these vectorspaces.
        '''
        pass


class FiniteLinearTransform(LinearTransform):
    ''' Class for linear transformation with
    finite-dimensional domain and co-domain.
    '''
    def __init__(self, domain: FiniteDimVectorSpace, 
                 co_domain: FiniteDimVectorSpace, field: Field):
        self.field = field
        self.domain = domain
        self.co_domain = co_domain
        super().__init__(domain, co_domain, field)

    def matrix_form(self) -> np.array:
        domain_basis = self.domain.get_std_basis()
        img = list(map(self.func, domain_basis))
        return self.co_domain.get_coordinates(img)
        
            


