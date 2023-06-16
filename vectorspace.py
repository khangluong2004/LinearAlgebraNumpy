import numpy as np
from field import *
from matrix import *
from abc import ABC, abstractmethod

class VectorSpace(ABC):
    ''' Abstract class of VectorSpace.'''
    def __init__(self, field: Field):
        self.field = field

    @abstractmethod
    def check_member(self, aug_vecs) -> bool:
        pass

    @abstractmethod
    def vec_add(self, a, b):
        pass

    @abstractmethod
    def scalar_mult(self, a: Field, b):
        pass


    def check_finite_subspace(self, test_set: set):
        ''' Check FINITE test_set of vectors on FINITE field 
        if it is a subspace'''

        # 0. Check all vectors in test_set is
        # in vectorspace
        for vec in test_set:
            if not self.check_member(vec):
                return(False)

        # 1. Non-empty
        if len(test_set) == 0:
            return False
        
        # 2. Closed under vector addition
        for vec1 in test_set:
            for vec2 in test_set:
                if self.vec_add(vec1, vec2) not in test_set:
                    return False
        
        # 3. Closed under scalar multiplication
        for scalar in self.field.elements:
            for vec in test_set:
                if self.scalar_mult(scalar, vec) not in test_set:
                    return False
        
        return True

    
class FiniteDimVectorSpace(VectorSpace):
    ''' Abstract class for finitely-dimensional vectorspace'''

    @abstractmethod
    def get_std_basis(self):
        pass

    def check_member(self, aug_vecs: list) -> bool:
        ''' aug_vecs: List of all vectors
        Return True if all aug_vec is element '''
        try:
            self.get_coordinates(aug_vecs)
            return True
        except ValueError as e:
            print(e)
            return False

    @abstractmethod
    def get_coordinates(self, aug_vecs: list) -> np.array:
        ''' Get coordinates from the finite basis
        aug_vecs: A list of vectors.
        Return the np.array matrix with each column
        = coordinates of each element.
        '''
        pass

    def from_coordinates_std(self, coord: np.array):
        ''' coord: A column n x 1 vectors'''
        std_basis = self.get_std_basis()
        dim = len(std_basis)
        vec = self.scalar_mult(coord[0][0], std_basis[0])
        for i in range(1, dim):
            vec = self.vec_add(self.scalar_mult(coord[i][0], std_basis[i]), vec)
        return vec

    
    def check_linear_independent(self, test_set: list) -> bool:
        ''' Check FINITE test_set on FINITE-dimensional vectorspace
        Note: test_set is a list with unique np.array 1 x n vector'''

        std_basis = self.get_std_basis()

        # 1. Dimension check: Linear independent set has at most 
        # dim(vectorspace) elements
        if len(test_set) > len(std_basis):
            return False
        
        # 2. Check if rank(aug_matrix) < len(test_set)
        aug_matrix = self.get_coordinates(test_set)
        rank = Matrix(aug_matrix, self.field).rank()
        if rank < len(test_set):
            return False
        return True

            
    
    def check_spanning_set(self, test_set: list) -> bool:
        ''' Check FINITE test_set on FINITE-dimensional vectorspace
        Note: test_set is a list with unique np.array n x 1 vector '''

        std_basis = self.get_std_basis()

        # 1. Dimension check: Spanning set has at least 
        # dim(vectorspace) elements
        if len(test_set) < len(std_basis):
            return False
        
        # 2. Check if rank(aug_matrix) < dim(vectorspace)
        aug_matrix = self.get_coordinates(np.concatenate(test_set, axis=1))
        rank = Matrix(aug_matrix, self.field).rank()
        if rank < len(std_basis):
            return False
        return True
    
    def extended_basis(self, lin_ind_set: list, 
                       sub_space_basis: list = None) -> list:
        ''' Extend a linearly independent set to become a basis
        of a subspace.

        lin_ind_set: list of vectors;
        sub_space_basis: list of each basis element;
        Vectors represented as column matrix (if possible)

        If no subspace basis specified, then current vectorspace
        standard basis is used.'''
        
        if sub_space_basis == None:
            sub_space_basis = self.get_std_basis()
            sub_space_coord = np.identity(len(sub_space_basis))
        else:
            sub_space_coord = self.get_coordinates(sub_space_basis)
        
        
        lin_ind_coord = self.get_coordinates(lin_ind_set)
        aug_matrix = np.concatenate((lin_ind_coord, sub_space_coord), axis=1)

        aug_obj = Matrix(aug_matrix, self.field)
        leading_post = aug_obj.leading_entry_position()
        chosen_sub_basis = [sub_space_basis[post[1] - len(lin_ind_set)] 
                            for post in leading_post if post[1] >= len(lin_ind_set)]
        return lin_ind_set + chosen_sub_basis

    
    def basis_from_spanning(self, span_set: list) -> list:
        ''' Extract the basis from spanning set.
        span_set: List  of n x 1 np.array vectors.
        Make sure span_set is a spanning set using check.'''

        span_set_coord = self.get_coordinates(span_set)
        aug_obj = Matrix(span_set_coord, self.field)
        leading_posts = aug_obj.leading_entry_position()
        return [span_set[post[1]] for post in leading_posts]

# Implementations
# Vector space R-n
class F_n(FiniteDimVectorSpace):
    def __init__(self, field: Field, dim: int):
        super().__init__(field)
        self.dim = dim
        self.std_basis = []
        for i in range(dim):
            curr = np.zeros((1, dim))
            curr[0][i] = 1
            self.std_basis.append(curr)
    
    def check_member(self, aug_vecs: list) -> bool:
        ''' All vectors must be 1 x n np.array of type real'''
        for vec in aug_vecs:
            col, row = vec.shape
            if col != 1 or row != self.dim:
                return False
            for elem in vec[0]:
                if not self.field.check_member(elem):
                    return False
        return True
    
    def vec_add(self, a: np.array, b: np.array) -> np.array:
        ''' a, b: 1 x n np.array vector'''
        return a + b
    
    def scalar_mult(self, a: Field, b: np.array) -> np.array:
        return a * b
    
    def get_std_basis(self):
        return self.std_basis
    
    def get_coordinates(self, aug_vecs: list) -> np.array:
        return np.transpose(np.concatenate(aug_vecs, axis=0))

# Vector spaces associated with a matrix
# For theses vector spaces, each vector is a 1 x n np.array vector 
# represented as 2D array: e.g: [[1, 2, 3]], not [1, 2, 3].
class RowSpace(FiniteDimVectorSpace):
    def __init__(self, field: Field, matrix_2d: list):
        super().__init__(field)
        self.matrix = np.array(matrix_2d)

        # Find std_basis
        matrix_obj = Matrix(self.matrix, self.field)
        matrix_obj.ref()
        _ , col = matrix_obj.matrix.shape
        result = []
        zeroes = np.zeros((1, col))
        for row in matrix_obj.matrix:
            if np.any(row != zeroes):
                result.append(row)
        self.std_basis = np.array(result)

    def get_coordinates(self, aug_vecs: list) -> np.array:
        ''' aug_vecs: List of column 1 x n vectors for this subspace.
        Return the matrix where each column is the corresponding
        vector coordinates.'''

        aug_vecs = np.transpose(np.concatenate(aug_vecs, axis=0))
        std_basis = self.std_basis
        aug_matrix = np.concatenate((np.transpose(std_basis), aug_vecs), axis=1)
        aug_obj = Matrix(aug_matrix, self.field)
        aug_obj.rref()
        
        _, input_col = aug_vecs.shape
        dim, _ = std_basis.shape
        zero_row = np.zeros((1, dim))
        result = np.zeros((dim, input_col))
        non_elem = set()
        for row in aug_obj.matrix:
            check = row[:dim]
            res = row[dim:]
            if np.any(check != zero_row):
                leading_entry = np.where(check != 0)[0][0]
                result[leading_entry] = res
            else:
                non_zeroes = np.where(res != 0)[0]
                if len(non_zeroes) != 0:
                    non_elem |= set(non_zeroes)
        if len(non_elem) != 0:
            raise ValueError(f"{non_elem} vectors is not in vectorspace")
        return(result)
    
    def vec_add(self, a: np.array, b: np.array) -> np.array:
        ''' For row/col/solution space, 
        vector add = field add on each element.
        Thus, for np.array, vector add = field add'''
        return self.field.add(a, b)
    
    def scalar_mult(self, a: Field, b: np.array) -> np.array:
        return self.field.mult(a, b)
    
    def get_std_basis(self) -> np.array:
        ''' Return a 2D np array (immutable)'''
        return self.std_basis
    
       
        

class ColSpace(RowSpace):
    def __init__(self, field: Field, matrix_2d: list):
        super().__init__(field, matrix_2d)

        # Find std_basis
        matrix_obj = Matrix(self.matrix, self.field)
        matrix_obj.ref()
        row , col = matrix_obj.matrix.shape
        result = []
        zeroes = np.zeros((1, col))
        for row in matrix_obj.matrix:
            if np.any(row != zeroes):
                leading_entry = np.where(row != 0)[0][0]
                result.append(self.matrix[:, leading_entry])
        self.std_basis = np.array(result)

class SolutionSpace(RowSpace):
    def __init__(self, field: Field, matrix_2d: list):
        super().__init__(field, matrix_2d)

        # Find std_basis
        matrix_obj = Matrix(self.matrix, self.field)
        matrix_obj.rref()
        num_row , num_col = matrix_obj.matrix.shape
        leading_entries = set()
        zeroes = np.zeros((1, num_col))
        for row in matrix_obj.matrix:
            if np.any(row != zeroes):
                leading_entries.add(np.where(row != 0)[0][0])
        
        # Save leading entries. Useful attributes for linear code, 
        # but not required
        self.leading_entries = leading_entries
        
        result = []
        for j in range(num_col):
            if j not in leading_entries:
                non_pivot = self.scalar_mult(-1, matrix_obj.matrix[:, j])
                non_pivot[j] = 1
                result.append(np.pad(non_pivot, pad_width=(0, num_col - num_row), constant_values=0.0))
        self.std_basis = np.array(result)
    
    def check_member(self, aug_vecs: np.array) -> bool:
        test = self.matrix @ aug_vecs
        return np.all(test != np.zeros(test.shape))


# Testing

# real = RealField()
# f5 = PrimeField(5)
# R2 = F_n(real, 2)
# sol_space = SolutionSpace(real, [[1, 2, 3], [0, 0, 3]])
# row_space = RowSpace(real, [[1, 2, 3], [0, 2, 3], [0, 0, 3]])
# col_space = ColSpace(real, [[1, 2, 3], [0, 2, 3], [0, 0, 3]])

# print(R2.get_std_basis())
# print(col_space.extended_basis([np.array([[0, 0, 3]])]))
# print(row_space.basis_from_spanning([np.array([[0, 0, 3]]), 
#                                      np.array([[1, 2, 3]]), 
#                                      np.array([[0, 2, 3]]),
#                                      np.array([[-1, -2, -3]])]))
# print(sol_space.get_std_basis())