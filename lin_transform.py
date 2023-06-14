import numpy as np
import math
from abc import abstractmethod, ABC
from vectorspace import *
from field import *

class LinearTransform(ABC):
    ''' A linear transformation T: V -> W with
    V and W sharing the same F is any T that:
    1. T(a) + T(b) = T(a+b)
    2. k T(a) = T(ka) (k: scalar; a, b: vectors)

    Can't check for these 2 properties :D 
    This class only defines the basic of a transformation.
    Need to ensure the linear properties by hands.
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
    
    @abstractmethod
    def func_basis(self, a):
        ''' Functions mapping to the image of domain basis.
        Used to construct the general functions (a bit easier)'''
        pass

    def func(self, a):
        ''' a: 1 x n np.array vectors'''
        domain_basis = self.domain.get_std_basis()
        coord = self.domain.get_coordinates([a])
        vec = self.co_domain.scalar_mult(coord[0][0], self.func_basis(domain_basis[0]))
        for i in range(1, len(coord)):
            vec = self.co_domain.vec_add(vec, self.co_domain.scalar_mult(coord[i][0], self.func_basis(domain_basis[i])))
        return vec
    
# Simple Geometric Linear Transformation R2 -> R2: 
# Reflection, Rotation, Stretch, Shear

class LinearTransformationR2(FiniteLinearTransform):
    def __init__(self, field: Field):
        self.domain = R_n(field, 2)
        self.co_domain = R_n(field, 2)
        self.field = field

class ReflectionR2(LinearTransformationR2):
    ''' Reflection about y-axis'''

    def func_basis(self, a):
        ''' a: 1x2 np.array vector'''
        return np.array([[a[0][1], a[0][0]]])

class RotationR2(LinearTransformationR2):
    ''' Rotation clockwise about origin '''
    def __init__(self, field: Field, degree: int):
        super().__init__(field)
        self.degree = degree

    def func_basis(self, a):
        ''' a: 1x2 np.array vector'''
        if np.all(a == [[1, 0]]):
            return np.array([[math.cos(self.degree), math.sin(self.degree)]])
        return np.array([[-math.sin(self.degree), math.cos(self.degree)]])

class StretchR2(LinearTransformationR2):
    ''' Stretch along x or y-axis'''
    def __init__(self, field: Field, scale: float, x_axis: bool):
        super().__init__(field)
        self.x_axis = x_axis
        self.scale = scale
    
    def func_basis(self, a):
        ''' a: 1x2 np.array vector'''
        if self.x_axis:
            return np.array([[self.field.mult(self.scale, a[0][0]), a[0][1]]])
        else:
            return np.array([[a[0][0], self.field.mult(self.scale, a[0][1])]])

class ShearR2(LinearTransformationR2):
    ''' Shear along x or y-axis'''
    def __init__(self, field: Field, scale: float, x_axis: bool):
        ''' x_axis: Shear along x or not'''
        super().__init__(field)
        self.x_axis = x_axis
        self.scale = scale
    
    def func_basis(self, a):
        ''' a: 1x2 np.array vector'''
        if self.x_axis:
            return np.array([[self.field.add(a[0][0], self.field.mult(self.scale, a[0][1])), a[0][1]]])
        else:
            return np.array([[a[0][0], self.field.add(a[0][1], self.field.mult(self.scale, a[0][0]))]]) 




# Testing
# real = RealField()
# reflection = ReflectionR2(real)
# rotation = RotationR2(real, math.pi/2)
# stretch = StretchR2(real, 2, True)
# shear = ShearR2(real, 2, True)

# print(reflection.func(np.array([[1, 2]])))
# print(rotation.matrix_form())
            
        



        
            


