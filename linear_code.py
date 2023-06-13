from itertools import combinations
import numpy as np
from vectorspace import *
from field import *

class LinearCode(SolutionSpace):
    ''' 
    Class for linear code. 
    Note that: a linear code is a solution space
    for a set matrix with additional method to:
    1. Encode
    2. Decode
    3. Find distance between codewords and minimum
    distance
    '''

    def __init__(self, field: Field, matrix_2d: list):
        super().__init__(field, matrix_2d)
        
        # Store computation expensive minimum distance
        self.min_distance = None
    
    def encode(self, a: np.array) -> np.array:
        ''' 
        a: n x 1 np.array vector containing;
        Return m x 1 np.array.
        '''
        return a @ self.get_std_basis()
    
    def decode(self, a: np.array, single_correct: bool = True) -> np.array:
        '''
        a: n x 1 np.array vector;
        Decode recieved message to obtain original info by:
        1. Check if valid codeword.
        2. If single_correct, fix 1 error by checking columns.
        3. Drop check bits.
        '''

        a = a.reshape(len(a), 1)
        check = self.matrix @ a
        _, num_col = self.matrix.shape

        # Correcting
        if np.any(check != 0):
            if not single_correct:
                raise ValueError("Not a valid codeword")
            else:
                found = False
                for i in range(num_col):
                    cur_col = np.transpose(self.matrix[:, i])
                    if np.all(self.scalar_mult(check[0], cur_col) == self.scalar_mult(cur_col[0], check)):
                        mult = self.field.inv_mult(cur_col[0], check[0])
                        neg_mult = self.field.mult(-1, mult)
                        a[i][0] = self.field.add(a[i][0], neg_mult)
                        found = True
                        break
                if found == False:
                    raise ValueError("More than one single error")
        

        decoded = [a[i][0] for i in range(num_col) if i not in self.leading_entries]
        return np.array(decoded)


    
    def hamming_distance(self, a: np.array, b: np.array) -> int:
        ''' a, b: n x 1 np.array vector/ sequence '''
        neg_b = self.scalar_mult(-1, b)
        check = self.vec_add(a, neg_b)
        return sum(check != 0)
    

    def minimum_distance(self):
        ''' Find minimum distance by finding the
        smallest linear dpendent set of columns'''
        if self.min_distance != None:
            return self.min_distance

        col_set = set()
        _, num_col = self.matrix.shape
        for i in range(num_col):
            col_set.add(tuple(self.matrix[:, i]))
        col_space_obj = ColSpace(self.field, self.matrix)
        for i in range(1, len(col_set) + 1):
            curr_comb = combinations(col_set, i)
            for curr_set in curr_comb:
                check = [np.array(temp).reshape(-1, 1) for temp in curr_set]
                if not col_space_obj.check_linear_independent(check):
                    self.min_distance = len(curr_set)
                    return len(curr_set)
        # If can't find linear dependent set, the only 
        # codeword is 0 vector. Return 0 then.
        return(0)

    def max_error_detected(self):
        if self.min_distance == None:
            self.minimum_distance()
        return(self.min_distance - 1)

    def max_error_fixable(self):
        if self.min_distance == None:
            self.minimum_distance()
        return((self.min_distance - 1) // 2)


        

# Testing
# real = RealField()
# lc = LinearCode(real, np.array([[1, 2, 3], [0, 0, 3]]))
# print(lc.get_std_basis())
# print(lc.decode(np.array([-4, 2, 0])))
# print(lc.minimum_distance())

