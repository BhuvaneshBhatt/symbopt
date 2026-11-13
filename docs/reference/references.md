# Scholarly references

The package combines ideas from nonlinear optimization, convex optimization, computational algebra, polynomial optimization, semidefinite programming, moment theory, and verified numerics. The following references provide the mathematical background for the implemented methods.

## Nonlinear and constrained optimization

- Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*, 2nd ed. Springer, 2006.
- Dimitri P. Bertsekas. *Nonlinear Programming*, 3rd ed. Athena Scientific, 2016.
- Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004.

These cover first- and second-order optimality conditions, KKT conditions, constraint qualifications, convex quadratic optimization, and duality.

## Linear optimization

- Dimitris Bertsimas and John N. Tsitsiklis. *Introduction to Linear Optimization*. Athena Scientific, 1997.
- Alexander Schrijver. *Theory of Linear and Integer Programming*. Wiley, 1986.

These provide standard simplex, revised-simplex, duality, and polyhedral background.

## Computational algebra

- David A. Cox, John Little, and Donal O'Shea. *Ideals, Varieties, and Algorithms*, 4th ed. Springer, 2015.
- David A. Cox, John Little, and Donal O'Shea. *Using Algebraic Geometry*, 2nd ed. Springer, 2005.

These cover Gröbner bases, elimination, algebraic varieties, and polynomial system solving.

## Polynomial, SOS, and moment optimization

- Jean B. Lasserre. "Global Optimization with Polynomials and the Problem of Moments." *SIAM Journal on Optimization* 11(3), 796–817, 2001.
- Pablo A. Parrilo. *Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization*. PhD thesis, California Institute of Technology, 2000.
- Monique Laurent. "Sums of Squares, Moment Matrices and Optimization over Polynomials." In *Emerging Applications of Algebraic Geometry*, IMA Volumes in Mathematics and its Applications 149, Springer, 2009.
- Grigoriy Blekherman, Pablo A. Parrilo, and Rekha R. Thomas, eds. *Semidefinite Optimization and Convex Algebraic Geometry*. SIAM, 2013.

These are the primary background for SOS Gram representations, polynomial positivity relaxations, moment matrices, localizing matrices, hierarchy growth, and flatness/atomic recovery.

## Semidefinite programming

- Lieven Vandenberghe and Stephen Boyd. "Semidefinite Programming." *SIAM Review* 38(1), 49–95, 1996.
- Farid Alizadeh. "Interior Point Methods in Semidefinite Programming with Applications to Combinatorial Optimization." *SIAM Journal on Optimization* 5(1), 13–51, 1995.

These provide the standard primal/dual SDP framework and numerical background.

## Verified numerics and interval methods

- Ramon E. Moore, R. Baker Kearfott, and Michael J. Cloud. *Introduction to Interval Analysis*. SIAM, 2009.
- Fredrik Johansson. "Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic." *IEEE Transactions on Computers* 66(8), 1281–1292, 2017.

These motivate rigorous enclosures, branch-and-bound reasoning, dependency issues, and Arb real-ball arithmetic.

## How references are used in this documentation

Method pages cite this bibliography rather than duplicating full citations. The references explain the mathematical methods; implementation-specific behavior is defined by the package documentation and tests.
