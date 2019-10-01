# Nonlinear LMPC

This code runs the LMPC to compute a (local) optimal solution to the following miminum time optimal control problem.

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_MixedIntegerFormulation/readmeFigures/minimumTimeProblem.png" width="500" />
</p>

The optimal solution to the above control problem steers the dubins car from the starting point x_s to the terminal point x_F while avoiding an obstacle and satisfying input constraints. 

The left figure below show the closed-loop trajectory at convergence, the first feasible trajectory and the stored data from the previous iterations. We notice that the LMPC avoids the obstacle and almost saturates the input acceleration constraints (right figure), as expected from the (local) optimal solution to minimum time optimal control problem.

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_MixedIntegerFormulation/readmeFigures/iterationEvolution.png" width="420" />
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_MixedIntegerFormulation/readmeFigures/velocityAcceleration.png" width="420" />
</p>

## LMPC Key Idea and Files

We propose to solve the above minimum time optimal control problem iteratively. We perform the task repeatedly and we use the closed-loop data to synthesize the LMPC policy. This folder contains the following files:

1) main.py: this file runs the closed-loop simulations.
2) LMPC.py: the lmpc object is used to store the file and to construct the the safe set and value function approximation which define the LMPC (Set safeSetOption = 'timeVarying' for the LMPC from [1], set safeSetOption = 'spaceVarying' for the local LMPC from [2] and l='all', P='all' and  safeSetOption = 'all' to implement the LMPC from [3]).
3) FTOCP.py: the ftocp object solves a finite time optimal control problem given the initial condition x_t, a terminal contraint set and a terminal cost function.
4) plot.py: run this files after main.py to plot the closed-loop trajectory, the closed-loop iteration cost and the comparison for different LMPC policies synthesized with different numbers of data points.

## References

[1] Ugo Rosolia and Francesco Borrelli. "Nonlinear  Learning  Model  Predictive  Control  for Time  Optimal  Control  Problems" submitted (Available soon).

[2] Ugo Rosolia. "Learning Model Predictive Control: Theory and Application." PhD Thesis (Available soon).

[3] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)

