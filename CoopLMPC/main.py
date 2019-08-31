import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import copy
import pickle

import os, sys

def main():
	# Number of agents
	n_a = 2
	n_x = 4
	n_u = 2

	# Define system dynamics and cost for each agent
	A = [np.nan*np.ones((n_x, n_x)) for _ in range(n_a)]
	B = [np.nan*np.ones((n_x, n_u)) for _ in range(n_a)]
	A[0] = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
	A[1] = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
	B[0] = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
	B[1] = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
	Q = np.diag([1.0, 1.0, 1.0, 1.0]) #np.eye(2)
	R = np.diag([0.1, 0.1]) #np.array([[1]])

	# Linear inequalities
	Hx = 1.*np.vstack((np.eye(n_x), -np.eye(n_x)))
	gx = 10.*np.ones((2*n_x))
	Hu = 1.*np.vstack((np.eye(n_u), -np.eye(n_u)))
	gu = 1.*np.ones((2*n_u))

	# Initial Condition
	x0 = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	x0[0] = np.array([[0, 0, 0, 0]]).T
	x0[1] = np.array([[1, 0, 0, 0]]).T

	# Goal condition
	xf = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	xf[0] = np.array([[2, 0, 0, 0]]).T
	xf[1] = np.array([[-1, 0, 0, 0]]).T

	# Check to make sure all agent dynamics, inital, and goal states have been defined
	if np.any(np.isnan(A)) or np.any(np.isnan(B)):
		raise(ValueError('A or B matricies have nan values'))
	if np.any(np.isnan(x0)) or np.any(np.isnan(xf)):
		raise(ValueError('Initial or goal states have empty entries'))
	if Q.shape[0] != Q.shape[1] or len(np.diag(Q)) != n_x:
		raise(ValueError('Q matrix not shaped properly'))
	if R.shape[0] != R.shape[1] or len(np.diag(R)) != n_u:
		raise(ValueError('Q matrix not shaped properly'))

	# Initialize FTOCP object
	N_feas = 10
	ftocp  = [FTOCP(N_feas, A[i], B[i], 0.1*Q, R, Hx=Hx, gx=gx, Hu=Hu, gu=gu) for i in range(n_a)]

	# ====================================================================================
	# Run simulation to compute feasible solutions for all agents
	# ====================================================================================
	# Intermediate waypoint to ensure collision-free trajectory
	plot_init = True
	waypt = [np.array([[2, 1.5, 0, 0]]).T, np.array([[0, -1.5, 0, 0]]).T]

	xcl_feas = x0
	ucl_feas = [np.empty((n_u,1)) for _ in range(n_a)]

	for i in range(n_a):
		mode = 1
		print('Solving for initial feasible trajectory for agent %i' % (i+1))
		t = 0
		# time Loop (Perform the task until close to the origin)
		while True:
			xt = xcl_feas[i][:,t] # Read measurements

			if mode == 1:
				xg = waypt[i].reshape((n_x))
				tol = 0
			else:
				xg = xf[i].reshape((n_x))
				tol = 3

			ftocp[i].solve(xt, xf=xg, verbose = 0) # Solve FTOCP

			# Read input and apply it to the system
			ut = ftocp[i].get_u_opt()[:,0].reshape((n_u, 1))
			xtp1 = ftocp[i].model(xt.reshape((n_x, 1)), ut)
			ucl_feas[i] = np.append(ucl_feas[i], ut, axis=1)
			xcl_feas[i] = np.append(xcl_feas[i], xtp1, axis=1)

			if np.linalg.norm(xtp1-xg.reshape((n_x, 1)), ord=2)**2 <= 10**(-tol):
				if mode == 1:
					mode += 1
				else:
					break

			t += 1

		print('Agent %i trajectory length: %i' % (i+1, ucl_feas[i].shape[1]))
		# print np.round(np.array(xcl_feas[i]).T, decimals=2)
		# print np.round(np.array(ucl_feas[i]).T, decimals=2)

	if plot_init:
		plt.figure(1)
		for i in range(n_a):
			c = matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1)))
			plt.plot(xcl_feas[i][0,:], xcl_feas[i][1,:], '.', c=c)
			plt.xlabel('$x_1$')
			plt.ylabel('$x_2$')
		plt.show()
	# ====================================================================================

	sys.exit()

	# ====================================================================================
	# Run LMPC
	# ====================================================================================

	# Initialize LMPC object
	N_LMPC = 4 # horizon length
	ftocp_for_lmpc = FTOCP(N_LMPC, A, B, Q, R) # ftocp solve by LMPC
	lmpc = LMPC(ftocp_for_lmpc, CVX=True) # Initialize the LMPC decide if you wanna use the CVX hull
	lmpc.addTrajectory(xcl_feasible, ucl_feasible) # Add feasible trajectory to the safe set
	totalIterations = 10 # Number of iterations to perform

	# run simulation
	# iteration loop
	for it in range(0,totalIterations):
		xcl = [x0] # initialize system state at interation it
		ucl =[]
		xt = x0
		time = 0
		# time Loop (Perform the task until close to the origin)
		while np.dot(xt, xt) > 10**(-10):
			xt = xcl[time] # Read measurements

			lmpc.solve(xt, verbose = 0) # Solve FTOCP

			# Read input and apply it to the system
			ut = lmpc.uPred[:,0][0]
			ucl.append(copy.copy(ut))
			xcl.append(copy.copy(lmpc.ftocp.model(xcl[time], ut)))
			time += 1

		# print np.round(np.array(xcl).T, decimals=2) # Uncomment to print trajectory
		# Add trajectory to update the safe set and value function
		lmpc.addTrajectory(xcl, ucl)

	# =====================================================================================


	# ====================================================================================
	# Compute optimal solution by solving a FTOCP with long horizon
	# ====================================================================================
	N = 500 # Set a very long horizon to fake infinite time optimal control problem
	ftocp_opt = FTOCP(N, A, B, Q, R)
	ftocp_opt.solve(xcl[0])
	xOpt = ftocp_opt.xPred
	uOpt = ftocp_opt.uPred
	costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
	print "Optimal cost is: ", costOpt[0]
	# Store optimal solution in the lmpc object
	lmpc.optCost = costOpt[0]
	lmpc.xOpt    = xOpt

	# Save the lmpc object
	filename = 'lmpc_object.pkl'
	filehandler = open(filename, 'w')
	pickle.dump(lmpc, filehandler)

if __name__== "__main__":
  main()
