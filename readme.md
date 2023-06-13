# Linear Algebra Automation
## Ongoing dev

Wrapping up MAST10022 Linear Algebra course (2023) by (trying to) automate algorithms involved in the subject.

Abstract classes used for blueprint of certain concepts, while common cases are implemented. 

Topics covered:
1. Field: Implemented Finite Prime, Real and Complex Field.
2. Matrix (coupled with Field): All 3 row operations, Gaussian elimination, ref/rref form, rank, det, inverse supported for any field.
3. Vectorspace: Abstract of VectorSpace + FiniteDimensionalVectorSpace. 
Supported check_finite_subspace, check_member, check_linear_indpendent, check_spanning_set, extended_basis(nc), basis_from_spanning (nc). Implemented RowSpace, ColSpace, and SolutionSpace.
4. Linear Code: Support encode, decode (auto fixing at max 1 error), find hamming_distance, minimum_distance, max_error_detected, max_error_fixable.
5. Linear Transformation: TBC