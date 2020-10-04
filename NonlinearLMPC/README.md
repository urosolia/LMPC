# LMPC for nonlinear systems

This folder contains two implementations of the LMPC for nonlinear minimum time optimal control problems.

## Mixed Integer Formulation

Tolder "SafeSetQfunction_MixedIntegerFormulation" contains a LMPC implementation which uses either the time-varying sampled safe set [1], the local sampled safe set [2] or the sampled safe set [3]. At each time step, the control action is computed after solving a mixed integer nonlinear programming. The algorthm handles the integer variables and uses IPOPT to solve the nonlinear programming. More details on the strategy used to handle the integer variables can be found in [3, Section V.2)].

## Nonlinear Formulation

The folder "SafeSetQfunction_NonlinearFormulation" contains a LMPC implementation which used the time-varying convex safe set [1]. The resulting time-varying LMPC is reformulated as a nonlinear programming which is solved using IPOPT.

## References 

[1] Ugo Rosolia and Francesco Borrelli. "Nonlinear  Learning  Model  Predictive  Control  for Time  Optimal  Control  Problems" submitted (Available soon).

[2] Ugo Rosolia. "Learning Model Predictive Control: Theory and Application." PhD Thesis (Available soon).

[3] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)
