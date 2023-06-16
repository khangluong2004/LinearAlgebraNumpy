# Linear Algebra Automation
## Ongoing dev

Wrapping up MAST10022 Linear Algebra course (2023) by (trying to) automate algorithms involved in the subject.

Dependencies:
1. numpy

Abstract classes used for blueprint of certain concepts, while common cases are implemented. 

Topics covered:
1. Field: Implemented Finite Prime, Real and Complex Field.
2. Matrix (coupled with Field): All 3 row operations, Gaussian elimination, ref/rref form, rank, det, inverse supported for any field.
3. Vectorspace: Abstract of VectorSpace + FiniteDimensionalVectorSpace. 
Supported check_finite_subspace, check_member, check_linear_indpendent, check_spanning_set, extended_basis, basis_from_spanning. Implemented RowSpace, ColSpace, and SolutionSpace.
4. Linear Code: Support encode, decode (auto fixing at max 1 error), find hamming_distance, minimum_distance, max_error_detected, max_error_fixable.
5. Linear Transformation: Abstraction + Implementations of simple geometric linear transformation 
(stretch, reflect, shear, rotate): R2 -> R2
6. Eigenvalues, eigenvectors & eigenspace: TBC
7. Diagonalisation & Fast power of diagonalisable matrix: TBC
8. Inner product (including matrix form): TBC
9. Gramm-Schmidt & orthogonal projection: TBC
10. Linear Regression: Compute the linear combination of `given inputs` which produce outputs that minmize squared error with `given outputs`.  
11. Orthogonal diagonalisation (& Conic sections): TBC
12. Unitarily diagonalisation: TBC

Might cover as a bit unrelated :D
1. Euclidean R3 geometry: TBC