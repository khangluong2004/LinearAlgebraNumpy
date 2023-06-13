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

    def check_member(self, aug_vecs: np.array) -> bool:
        ''' aug_vecs: An augmented matrix created from
        vectors needing to check members in form:
        aug_vecs = [b1 |b2 |b3 ...] (where b1, b2, b3 are 
        1x n vectors). Faster computation. 
        Return True if all aug_vec is element '''
        try:
            self.get_coordinates(aug_vecs)
            return True
        except ValueError:
            return False

    @abstractmethod
    def get_coordinates(self, aug_vecs) -> np.array:
        ''' Get coordinates from the finite basis
        aug_vecs: A set of elements, or an np.array matrix
        if each element is a column matrix.
        Return the np.array matrix with each column
        = coordinates of each element.
        '''
        pass


    
    def check_linear_independent(self, test_set: list) -> bool:
        ''' Check FINITE test_set on FINITE-dimensional vectorspace
        Note: test_set is a list with unique np.array n x 1 vector'''

        std_basis = self.get_std_basis()

        # 1. Dimension check: Linear independent set has at most 
        # dim(vectorspace) elements
        if len(test_set) > len(std_basis):
            return False
        
        # 2. Check if rank(aug_matrix) < len(test_set)
        aug_matrix = self.get_coordinates(np.concatenate(test_set, axis=1))
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
                       sub_space_basis: np.array = None) -> np.array:
        ''' Extend a linearly independent set to become a basis
        lin_ind_set: list of n x 1 np.array vectors;
        sub_space_basis: 2D np.array with each basis element as a row;
        If no subspace basis specified, then current vectorspace
        standard basis is used.'''
          
        if sub_space_basis == None:
            sub_space_basis = np.transpose(self.get_std_basis())
        else:
            sub_space_basis = np.transpose(sub_space_basis)
        
        lin_ind_aug = np.concatenate(lin_ind_set, axis=1)
        aug_matrix = np.concatenate((lin_ind_aug, sub_space_basis), axis=1)

        aug_obj = ColSpace(self.field, aug_matrix)
        return aug_obj.get_std_basis()

    
    def basis_from_spanning(self, span_set: list) -> np.array:
        ''' Extract the basis from spanning set.
        span_set: List  of n x 1 np.array vectors.
        Make sure span_set is a spanning set using check.'''

        span_set_aug = np.concatenate(span_set, axis=1)
        aug_obj = ColSpace(self.field, span_set_aug)
        return aug_obj.get_std_basis()


# Vector spaces associated with a matrix
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

    def get_coordinates(self, aug_vecs: np.array) -> np.array:
        ''' aug_vecs: An augmented matrix created from
        vectors needing to find coordinates in form:
        aug_vecs = [b1 |b2 |b3 ...] (where b1, b2, b3 are 
        1x n vectors)
        Faster computation. '''

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
# sol_space = SolutionSpace(real, [[1, 2, 3], [0, 0, 3]])
# print(sol_space.get_std_basis())