import numpy as np
import numpy.linalg as la
import cvxpy as cp
import pdb
import multiprocessing as mp
import itertools

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import os, sys, time, copy, pickle

BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-1]))
sys.path.append(BASE_DIR)

from FTOCP_coop import FTOCP
from LMPC_coop import LMPC
import utils.plot_utils

pdb.set_trace()

def solve_init_traj(ftocp, x0, waypt, xf):
	n_x = ftocp.n
	n_u = ftocp.d

	xcl_feas = x0
	ucl_feas = np.empty((n_u,0))

	if waypt is None:
		mode = 1
	else:
		mode = 0
	t = 0

	# time Loop (Perform the task until close to the origin)
	while True:
		xt = xcl_feas[:,t] # Read measurements

		if mode == 0:
			xg = waypt.reshape((n_x))
			tol = 0
		else:
			xg = xf.reshape((n_x))
			tol = 5

		(x_pred, u_pred) = ftocp.solve(xt, xf=xg, CVX=True, verbose=False) # Solve FTOCP

		# Read input and apply it to the system
		ut = u_pred[:,0].reshape((n_u, 1))
		xtp1 = ftocp.model(xt.reshape((n_x, 1)), ut)
		ucl_feas = np.append(ucl_feas, ut, axis=1)
		xcl_feas = np.append(xcl_feas, xtp1, axis=1)

		# print('Time step: %i, Distance: %g' % (t, la.norm(xtp1-xf.reshape((n_x,1)), ord=2)))
		# Close within tolerance
		if la.norm(xtp1-xg.reshape((n_x,1)), ord=2) <= 10**(-tol):
			if mode == 1:
				break
			mode += 1

		t += 1

	return (xcl_feas, ucl_feas)

def solve_lmpc(lmpc, x0, xf, deltas, verbose=False, visualizer=None, pause=False):
	n_x = lmpc.ftocp.n
	n_u = lmpc.ftocp.d

	xcl = x0 # initialize system state at interation it
	ucl = np.empty((n_u,0))
	tol = 5

	t = 0
	# time Loop (Perform the task until close to the origin)
	while True:
		xt = xcl[:,t] # Read measurements
		(x_pred, u_pred) = lmpc.solve(xt, xf=xf, abs_t=t, deltas=deltas, verbose=verbose) # Solve FTOCP

		if visualizer is not None:
			visualizer.plot_state_traj(xcl, x_pred, t, deltas=deltas)
			visualizer.plot_act_traj(ucl, u_pred, t)

		# Read input and apply it to the system
		ut = u_pred[:,0].reshape((n_u, 1))
		xtp1 = lmpc.ftocp.model(xt.reshape((n_x, 1)), ut)

		ucl = np.append(ucl, ut, axis=1)
		xcl = np.append(xcl, xtp1, axis=1)

		# print('Time step: %i, Distance: %g' % (t, la.norm(xtp1-xf.reshape((n_x,1)), ord=2)))
		if la.norm(xtp1-xf.reshape((n_x,1)), ord=2) <= 10**(-tol):
			break

		if pause:
			raw_input('Iteration %i. Press enter to continue: ' % t)

		t += 1

	# print np.round(np.array(xcl).T, decimals=2) # Uncomment to print trajectory
	# Add trajectory to update the safe set and value function
	# lmpc.addTrajectory(xcl, ucl)
	return xcl, ucl

def get_traj_deltas(agent_xcls, xf, r_a=0):
	deltas = []
	n_a = len(agent_xcls)
	n_x = agent_xcls[0].shape[0]
	traj_lens = [agent_xcls[i].shape[1] for i in range(n_a)]
	max_traj_len = np.amax(traj_lens)

	# Pad trajectories so they are all the same length
	for i in range(n_a):
		if traj_lens[i] < max_traj_len:
			n_rep = max_traj_len - traj_lens[i]
			agent_xcls[i] = np.append(agent_xcls[i], np.tile(agent_xcls[i][:,-1].reshape((n_x,1)), n_rep), axis=1)

	# Solve for pair-wise distances between trajectory points for each agent at each time step
	for i in range(max_traj_len):
		# print('Timestep %i' % i)
		d = cp.Variable(n_a)
		pairs = map(list, itertools.combinations(range(n_a), 2))

		# Constraints
		constr = [d >= 0]
		for p in pairs:
			# r_i+r_j <= ||x_i-x_j||_2
			# print('d[%i] + d[%i] <= %g' % (p[0], p[1], la.norm(agent_xcls[p[0]][:2,i]-agent_xcls[p[1]][:2,i], ord=2)-2*r_a))
			constr += [d[p[0]]+d[p[1]] <= la.norm(agent_xcls[p[0]][:2,i]-agent_xcls[p[1]][:2,i], ord=2)-2*r_a]
			# constr += [d[p[0]]+d[p[1]] <= la.norm(agent_xcls[p[0]][:2,i]-agent_xcls[p[1]][:2,i], ord=np.inf)-2*r_a]

		# Cost
		if n_a == 2:
			w = np.array([min(i+1,traj_lens[j]) for j in range(n_a)])
		else:
			w = np.array([la.norm(agent_xcls[j][:2,i]-xf[j][:2], ord=2) for j in range(n_a)])
		cost = w*d

		problem = cp.Problem(cp.Maximize(cost), constr)
		try:
		    problem.solve()
		except Exception as e:
		    print(e)

		if problem.status == 'infeasible':
			raise(ValueError('Optimization was infeasible for step %i' % i))
		elif problem.status == 'unbounded':
			raise(ValueError('Optimization was unbounded for step %i' % i))

		deltas.append(d.value)

	deltas = np.array(deltas).T

	return deltas

def main():
	plot_dir = '/'.join((BASE_DIR, 'plots'))
	if not os.path.exists(plot_dir):
	    os.makedirs(plot_dir)

	log_dir = '/'.join((BASE_DIR, 'logs'))
	if not os.path.exists(log_dir):
	    os.makedirs(log_dir)
		
	# Flags
	parallel = False # Parallelization flag
	plot_init = False # Plot initial trajectory
	pause_each_solve = False # Pause on each FTOCP solution

	# Number of agents
	n_a = 3
	n_x = 4
	n_u = 2

	r_a = 0.1 # Agents are circles with radius a_r

	# Define system dynamics and cost for each agent
	A = [np.nan*np.ones((n_x, n_x)) for _ in range(n_a)]
	B = [np.nan*np.ones((n_x, n_u)) for _ in range(n_a)]
	A[0] = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
	A[1] = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
	A[2] = np.array([[1, 0, 0.2, 0],[0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
	B[0] = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
	B[1] = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
	B[2] = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
	Q = np.diag([1.0, 1.0, 1.0, 1.0]) #np.eye(2)
	R = np.diag([0.1, 0.1]) #np.array([[1]])

	# Linear inequalities
	Hx = 1.*np.vstack((np.eye(n_x), -np.eye(n_x)))
	gx = 10.*np.ones((2*n_x))
	Hu = 1.*np.vstack((np.eye(n_u), -np.eye(n_u)))
	gu = 2.*np.ones((2*n_u))

	# Initial Condition
	x0 = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	x0[0] = np.array([[-1, 1, 0, 0]]).T
	x0[1] = np.array([[1, 1, 0, 0]]).T
	x0[2] = np.array([[2, 0, 0, 0]]).T

	# Goal condition
	xf = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	xf[0] = np.array([[1, -1, 0, 0]]).T
	xf[1] = np.array([[-1, -1, 0, 0]]).T
	xf[2] = np.array([[-2, 0, 0, 0]]).T

	# Check to make sure all agent dynamics, inital, and goal states have been defined
	if np.any(np.isnan(A)) or np.any(np.isnan(B)):
		raise(ValueError('A or B matricies have nan values'))
	if np.any(np.isnan(x0)) or np.any(np.isnan(xf)):
		raise(ValueError('Initial or goal states have empty entries'))
	if Q.shape[0] != Q.shape[1] or len(np.diag(Q)) != n_x:
		raise(ValueError('Q matrix not shaped properly'))
	if R.shape[0] != R.shape[1] or len(np.diag(R)) != n_u:
		raise(ValueError('Q matrix not shaped properly'))

	# ====================================================================================
	# Run simulation to compute feasible solutions for all agents
	# ====================================================================================
	# Intermediate waypoint to ensure collision-free trajectory
	# waypt = [np.array([[2, 1.5, 0, 0]]).T, np.array([[0, -1.5, 0, 0]]).T]
	waypt = None

	xcl_feas = []
	ucl_feas = []

	# Initialize FTOCP objects
	N_feas = 10
	ftocp = [FTOCP(N_feas, A[i], B[i], 0.1*Q, R, Hx=Hx, gx=gx, Hu=Hu, gu=gu) for i in range(n_a)]

	start = time.time()
	if parallel:
		# Create threads
		pool = mp.Pool(processes=n_a)
		# Assign thread to agent trajectory
		results = [pool.apply_async(solve_init_traj, args=(ftocp[i], x0[i], None, xf[i])) for i in range(n_a)]
		# Sync point
		init_trajs = [r.get() for r in results]

		(xcl_feas, ucl_feas) = zip(*init_trajs)
		xcl_feas = list(xcl_feas)
		ucl_feas = list(ucl_feas)
	else:
		for i in range(n_a):
			(x, u) = solve_init_traj(ftocp[i], x0[i], None, xf[i])
			xcl_feas.append(x)
			ucl_feas.append(u)
	end = time.time()

	for i in range(n_a):
		xcl_feas[i] = np.append(xcl_feas[i], xf[i], axis=1)
		ucl_feas[i] = np.append(ucl_feas[i], np.zeros((n_u,1)), axis=1)

	xcl_lens = [xcl_feas[i].shape[1] for i in range(n_a)]

	for i in range(n_a):
		before_len = 0
		after_len = 0
		for j in range(i):
			before_len += xcl_lens[j]
		for j in range(i+1, n_a):
			after_len += xcl_lens[j]

		xcl_feas[i] = np.hstack((np.tile(x0[i], before_len), xcl_feas[i], np.tile(xf[i], after_len)))
		ucl_feas[i] = np.hstack((np.zeros((n_u, before_len)), ucl_feas[i], np.zeros((n_u, after_len))))

	# pdb.set_trace()

	print('Time elapsed: %g s' % (end - start))

	if plot_init:
		plot_utils.plot_agent_trajs(xcl_feas, r=r_a, trail=True)

	# ====================================================================================

	# ====================================================================================
	# Run LMPC
	# ====================================================================================

	# Initialize LMPC objects for each agent
	N_LMPC = [6, 6, 6] # horizon lengths
	ftocp_for_lmpc = [FTOCP(N_LMPC[i], A[i], B[i], Q, R, Hx=Hx, gx=gx, Hu=Hu, gu=gu) for i in range(n_a)]# ftocp solve by LMPC
	lmpc = [LMPC(f, CVX=False) for f in ftocp_for_lmpc]# Initialize the LMPC decide if you wanna use the CVX hull
	for i in range(n_a):
		print('Agent %i' % (i+1))
		lmpc[i].addTrajectory(xcl_feas[i], ucl_feas[i], xf[i]) # Add feasible trajectory to the safe set

	xcls = [copy.copy(xcl_feas)]
	ucls = [copy.copy(ucl_feas)]
	delta_log = []

	totalIterations = 20 # Number of iterations to perform
	start_time = str(time.time())
	exp_dir = '/'.join((plot_dir, start_time))
	os.makedirs(exp_dir)

	# Initialize objective plots
	# obj_plot = plot_utils.updateable_plot(n_a, title='Agent Trajectory Costs', x_label='Iteration')
	# delta_plot = plot_utils.updateable_ts(n_a, title='Deltas', x_label='Time', y_label=['Agent %i' % (i+1) for i in range(n_a)])
	lmpc_vis = [plot_utils.lmpc_visualizer(pos_dims=[0,1], n_state_dims=n_x, n_act_dims=n_u, agent_id=i, plot_dir=exp_dir) for i in range(n_a)]

	raw_input('Ready to run LMPC, press enter to continue...')

	# run simulation
	# iteration loop
	for it in range(totalIterations):
		for lv in lmpc_vis:
			lv.update_prev_trajs(state_traj=xcls[-1], act_traj=ucls[-1])

		deltas = get_traj_deltas(xcls[-1], xf, r_a=r_a) # Compute deltas with last trajectory
		delta_log.append(deltas)
		# delta_plot.update(deltas)

		f = plot_utils.plot_agent_trajs(xcls[-1], r=r_a, deltas=deltas, trail=False)
		pdb.set_trace()
		f.savefig('/'.join((plot_dir, start_time, 'it_%i_trajs.png' % it)))

		x_it = []
		u_it = []
		for i in range(n_a):
			print('Agent %i' % (i+1))
			(xcl, ucl) = solve_lmpc(lmpc[i], x0[i], xf[i], deltas[i,:], visualizer=lmpc_vis[i], pause=pause_each_solve)
			opt_cost = lmpc[i].addTrajectory(xcl, ucl)
			# obj_plot.update(np.array([it, opt_cost]).T, i)
			x_it.append(xcl)
			u_it.append(ucl)

		xcls.append(x_it)
		ucls.append(u_it)
	# =====================================================================================


	# ====================================================================================
	# Compute optimal solution by solving a FTOCP with long horizon
	# ====================================================================================
	# N = 500 # Set a very long horizon to fake infinite time optimal control problem
	# ftocp_opt = FTOCP(N, A, B, Q, R)
	# ftocp_opt.solve(xcl[0])
	# xOpt = ftocp_opt.xPred
	# uOpt = ftocp_opt.uPred
	# costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
	# print "Optimal cost is: ", costOpt[0]
	# # Store optimal solution in the lmpc object
	# lmpc.optCost = costOpt[0]
	# lmpc.xOpt    = xOpt

	# Save the lmpc object
	filename = '/'.join((log_dir, 'lmpc_object.pkl'))
	filehandler = open(filename, 'w')
	pickle.dump(lmpc, filehandler)

	plt.show()

if __name__== "__main__":
  main()
