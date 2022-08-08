# Optimization visualizer

The goal of this project is to visualize how different optimization algorithms behave in both convex and non-convex 2D functions.

There are two main areas of focus: Line search and trust-region methods. 

Currently search direction has been implemented with steepest descent and newton's method with a backtracking line search which satisfies the armijo conditions being used to find the step size. 

In the cases of the newton's method - if the Hessian is indefinite there is an implementation to force it to become positive definite using addition of multiple identity or modifying the negative eigenvalues. This modification is useful in the case of the Himmelblau function when the newton's method fails since the Hessian is indefinite or negative definite. 

The derivatives of the functions are found using the finite difference method so theoretically any 2D function could be visualized here. 

Currently implementing the trust-region methods which are the cauchy and dog-leg.
