# Linear LMPC

This code runs the LMPC from [1] and [2] to solve the following Contratined LQR problem

<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/LinearLMPC/readmeFigures/CLQR.png" width="500" />
</p>

The LMPC will improve the closed-loop performance, unitl the closed-loop trajectory converges to a steady state behavior. This state state closed-loop trajectory is the unique gloabl optimal solution to above control problem, if some the technical conditions hold. For more details we refer to [1].
<p align="center">
<img src="https://github.com/urosolia/LMPC/blob/master/LinearLMPC/readmeFigures/closed-loop.png" width="420" />
<img src="https://github.com/urosolia/LMPC/blob/master/LinearLMPC/readmeFigures/costImprovement.png" width="420" />
</p>

## LMPC Key Idea and Files

We propose to solve the above CLQR problem iteratively. We perform the regulation task repeatedly and we use the closed-loop data to synthesize the LMPC policy. This folder contains the following files:

1) main.py: this file runs the closed-loop simulations.
2) LMPC.py: the lmpc object is used to store the file and to construct the the safe set and value function approximation which define the LMPC policy (Set CVX = False for the LMPC from [1] and CVX = True for the LMPC from [2]).
3) FTOCP.py: the ftocp object solves a finite time optimal control problem given the initial condition x_t, a terminal contraint set (optional) and a terminal cost function (optional)
4) plot.py: run this files after main.py to plot the closed-loop trajectory, the closed-loop iteration cost and the comparison with the optimal solution to the CLQR problem.

## References

[1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)

[2] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017). [PDF](https://www.sciencedirect.com/science/article/pii/S2405896317306523)