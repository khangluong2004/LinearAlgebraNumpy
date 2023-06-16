import numpy as np
from field import *
from matrix import *
from vectorspace import *

class LinearRegression:
    ''' 
    Field: R_n; 

    Finding the line/ polynomial curve of best fit
    with minimal square error through list of given
    coordinates: [(x1, y1), (x2, y2), ...].
    Let A = [[1 x1 x1^2 ...], [1 x2 x2^2 ...], ...];
    y = [[y1 y2 y3 ...]]^T; 
    coeff_vec = [a0 a1 a2 ...], such that 
    best curve = a0 + a1 * x1 + a2 * x2 + ...

    Method implemented works by finding the projection of
    y onto the vector space {Ax | All col vec x}.
    Equivalent to solving equation for u: A^T * Au = A^T * y.
    Row op. & rref form would be sufficient.

    Generalisation is provided to support multiple inputs
    mapped to 1 output (x1, z1, ...). 
    Note that, for multiple inputs, the result would
    be the coeff for an optimal linear combinations of the 
    input, which may not be a line neccessarily.
    '''
    def __init__(self, input_dim: int):
        ''' input_dim: The dimension of the inputs 
        e.g: dim = 2 => (x1, z1) -> y1;
        '''
        self.input_dim = input_dim
    
    def best_fit_coeff(self, x_coords: list, y_coords: list):
        '''
        x_coords: input_dim x n list : 
        [[x1, z1, ...], [x2, z2, ...],...];

        y_coords: 1 x n list: [y1, y2, ...]

        Must match in dimension
        '''
        input_dim = self.input_dim
        y_coords = np.array(y_coords).reshape(-1, 1)
        A = np.array(x_coords)
        
        
        aug_matrix = np.concatenate((np.transpose(A) @ A, np.transpose(A) @ y_coords), axis=1)
        aug_obj = Matrix(aug_matrix, RealField())
        aug_obj.rref()
        
        
        last_col = aug_obj.matrix[:, -1]
        coeff = np.zeros((input_dim + 1, 1))
        leading_entries = aug_obj.leading_entry_position()
        for leading in leading_entries:
            coeff[leading[1]][0] = last_col[leading[0]]
        return coeff

class LinRegPolynomial(LinearRegression):
    ''' An extension to adapt to polynomial on R2.
    input_dim = Degree of polynomial '''

    def best_fit_coeff(self, x_coords: list, y_coords: list):
        ''' 
        x_coords: 1 x n list [x1, x2, ...]
        y_coords: 1 x n list [y1, y2, ...]
        Must match in dimension
        '''
        new_x_coords = [0] * len(x_coords)
        for i in range(len(x_coords)):
            curr = [0] * (self.input_dim + 1)
            for j in range(self.input_dim + 1):
                curr[j] = x_coords[i] ** j
            new_x_coords[i] = curr
        return super().best_fit_coeff(new_x_coords, y_coords)

    
    


# Testing
# lin_reg = LinearRegression(1)
# poly_reg = LinRegPolynomial(2)
# print(lin_reg.best_fit_coeff([[1, -1], [1, 1], [1, 2], [1, 4]], [1, 1, 3, 5]))
# print(poly_reg.best_fit_coeff([-1, 1, 2, 4], [1, 1, 3, 5]))
