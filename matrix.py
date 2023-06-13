import numpy as np
from field import *

class Matrix:
    ''' Numpy matrix with row operations, 
    row reduced, inverse and determinant methods.
    Customised field. 
    Built-in field: R and Prime Finite Field.'''

    def __init__(self, matrix_2d: list, field: Field = None):
        self.matrix = np.array(matrix_2d, dtype=float)
        if field == None:
            self.field = RealField()
        else:
            self.field = field

    def __str__(self):
        return(f" Field (None = R): {self.field} \n Matrix: \n {self.matrix}")

    # Row operations 
    def swap(self, r1: int, r2: int) -> None:
        ''' Perform in-place row swap. 
            r1, r2 are 0-indexed. '''
        matrix = self.matrix
        temp = np.array(matrix[r1])
        matrix[r1] = matrix[r2]
        matrix[r2] = temp

    def add_row(self, r1: int, r2: int, 
                mult1: float = 1.0, mult2: float = 1.0) -> None:
        ''' Convert in-place r1 row = mult1 * r1 + mult2 * r2.'''
        matrix = self.matrix
        field = self.field
        if (mult1 == 0 or mult2 == 0):
            raise ValueError("Non-zero multiplier")
        
        matrix[r1] = field.add(field.mult(mult1, matrix[r1]), 
                                field.mult(mult2, matrix[r2]))

    def mult_row(self, r1: int, mult1: float) -> None:
        ''' In-place row multiples.'''
        matrix = self.matrix
        field = self.field
        if (mult1 == 0):
            raise ValueError("Non-zero multiplier")
        
        matrix[r1] = field.mult(mult1, matrix[r1])

    # Gaussian Elimination
    def ref(self) -> None:
        ''' In-place row reduced to row echelon form'''
        matrix = self.matrix
        field = self.field
        row, col = matrix.shape
        first_row = 0
        for i in range(col):
            cur_col = matrix[first_row:, i] 
            if np.any(cur_col != 0):
                non_zero_rows = np.where(cur_col != 0)[0]
                self.swap(first_row, first_row + non_zero_rows[0])

                swapped = non_zero_rows[0]
                cur_col = matrix[first_row:, i]
                for j in non_zero_rows:
                    if j == swapped:
                        continue
                    self.add_row(first_row + j, first_row, 
                                 cur_col[0], -cur_col[j])
                first_row += 1


    def rref(self) -> None:
        ''' In-place row reduced to reduced row echelon form'''
        matrix = self.matrix
        field = self.field
        row, col = matrix.shape
        first_row = 0
        for i in range(col):
            cur_col = matrix[first_row:, i]

            if np.any(cur_col != 0):
                non_zero_rows = np.where(cur_col != 0)[0]
                
                mult1 = field.inv_mult(1, cur_col[non_zero_rows[0]])
                self.mult_row(first_row + non_zero_rows[0], mult1) 
                self.swap(first_row, first_row + non_zero_rows[0])


                swapped = non_zero_rows[0]
                cur_col = matrix[first_row:, i]
                for j in non_zero_rows:
                    if j == swapped:
                        continue
                    self.add_row(first_row + j, first_row, 1, -cur_col[j])


                above_col = matrix[:first_row, i]
                start = 0
                for mult in above_col:
                    if mult != 0:
                        self.add_row(start, first_row, 1, -mult)
                    start += 1
                first_row += 1

    def rank(self) -> int:
        ''' Find rank of a matrix. '''
        matrix = Matrix(self.matrix, self.field)
        matrix.rref()
        return sum([1 for row in matrix.matrix if np.any(row != np.zeros(row.shape))])
    
    def leading_entry_position(self) -> list:
        ''' Get leading entry positions in rref form'''
        copy_obj = Matrix(self.matrix, self.field)
        copy_obj.rref()

        post = []
        for i in range(len(copy_obj.matrix)):
            row = copy_obj.matrix[i]
            check = np.where(row != 0)[0]
            if len(check) != 0:
                post.append((i, check[0]))
        
        return(post)


    # Finding inverse using Gaussian elimination
    def inv(self) -> np.array:
        ''' Find the inverse of n x n `matrix` using Gaussian elimination.
        Return None if no inverse exists. '''
        matrix = self.matrix
        field = self.field
        row, col = matrix.shape
        if row != col:
            raise ValueError("Not a square matrix")
        identity = np.identity(row)
        aug_obj = Matrix(np.concatenate((matrix, identity), axis=1), field)
        aug_obj.rref()
        aug_matrix = aug_obj.matrix
        if np.all(aug_matrix[:, :row] == identity):
            return(aug_matrix[:, -row:])
    
    # Finding determinant
    def det(self) -> float:
        ''' Find determinant of matrix (with respect to field)
        using co-factor expansion on 1st row '''
        matrix = self.matrix
        row, col = matrix.shape
        if row != col:
            raise ValueError("No determinant defined for non-square matrix")
        if row == 1:
            return matrix[0, 0]
        determinant = 0
        for i in range(col):
            minor = np.concatenate((matrix[1:, :i], matrix[1:, i+1:]), axis=1)
            determinant = self.field.add(determinant, 
                                         self.field.mult(self.field.mult((-1)**(1 + i + 1), matrix[0, i]), 
                                           Matrix(minor, self.field).det()))
        return(determinant)
    




# Testing
# f5 = PrimeField(5)
# matrix = Matrix([[0, 1, 0, 2], [0, 0, 0, 0]])
# print(matrix.leading_entry_position())