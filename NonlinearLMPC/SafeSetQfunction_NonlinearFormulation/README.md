# Nonlinear LMPC

This code runs the LMPC to compute a (local) optimal solution to the following miminum time optimal control problem.

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_NonlinearFormulation/readmeFigures/minimumTimeProblem.png" width="500" />
</p>

The optimal solution to the above control problem steers the dubins car along a curve of radius R, from the starting point x_s to the terminal point x_F in minimum time. We initialize the LMPC using the following first feasible trajectory.

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_NonlinearFormulation/readmeFigures/feasibleTrajectory.png" width="500" />
</p>

After few iterations the LMPC converges to a steady-state behavior which saturates both the road boundaries and the input acceleration constraints, as expected from the (local) optimal solution to minimum time optimal control problem.

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_NonlinearFormulation/readmeFigures/closedLoopTrajectory.png" width="420" />
<img src="https://github.com/urosolia/LMPC/blob/master/NonlinearLMPC/SafeSetQfunction_NonlinearFormulation/readmeFigures/inputAtConvergence.png" width="420" />
</p>

## LMPC Key Idea and Files

We propose to solve the above minimum time optimal control problem iteratively. We perform the task repeatedly and we use the closed-loop data to synthesize the LMPC policy. This folder contains the following files:

1) main.py: this file runs the closed-loop simulations.
2) LMPC.py: the lmpc object is used to store the file and to construct the the safe set and value function approximation which define the LMPC policy from [1].
3) FTOCP.py: the ftocp object solves a finite time optimal control problem given the initial condition x_t, a terminal contraint set and a terminal cost function.
4) plot.py: run this files after main.py to plot the closed-loop trajectory, the closed-loop iteration cost and the comparison for different LMPC policies synthesized with different numbers of data points.

## References

[1] Ugo Rosolia and Francesco Borrelli. "Nonlinear  Learning  Model  Predictive  Control  for Time  Optimal  Control  Problems" submitted (Available soon).
