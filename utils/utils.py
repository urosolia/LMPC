from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.spatial
import cvxpy as cp
import pdb, itertools

def get_agent_polytopes(A, abs_t, xf_reached, r_a):
	# A is a matrix with rows equal to number of agents and columns equal to
	# number of position dimensions
	n_a = A.shape[0]

	H_t = [[] for _ in range(n_a)]
	g_t = [[] for _ in range(n_a)]

	if n_a > 2:
		# Calculate Voronoi graph of agent positions at a timestep (only works
		# for 3 or more agents)
		vor = scipy.spatial.Voronoi(A)
		# Iterate through agent locations
		for (i, point) in enumerate(vor.points):
			# Initialize inequality matrix and vector (H_a*x <= g_a)
			H_a = []
			g_a = []
			# Ridges are the line segments or rays which separate two agents in
			# position space
			ridges = zip(vor.ridge_points.tolist(), vor.ridge_vertices)

			# Iterate through all ridges to find ones which include the current
			# agent location
			for r in ridges:
				rp = np.array(r[0]) # Agent location indicies
				rv = np.array(r[1]) # Vertex indicies of the ridge (-1 indicates unbounded)

				if i in rp:
					rp_idx = np.argwhere(rp == i)[0][0]

					tr_1 = (xf_reached[i]-abs_t)/xf_reached[i] # ratio of remaining trajectory for current agent
					tr_2 = (xf_reached[rp[1-rp_idx]]-abs_t)/xf_reached[i] # ratio of remaining trajectory for neighbor agent
					# shifted sigmoid between 0 (small t_r) and 1 (large t_r)
					w_1 = np.exp(35*tr_1+5)/(np.exp(35*tr_1+5)+1)
					w_2 = np.exp(35*tr_2+5)/(np.exp(35*tr_2+5)+1)

					v = [] # Vertices that define the ridge
					p = [point, vor.points[rp[1-rp_idx]]] # Points that the ridge separates
					for k in range(len(rv)):
						if rv[k] >= 0:
							v.append(vor.vertices[rv[k]])

					conn_vec_p = p[1] - p[0] # Vector from current point to second point
					conn_vec_l = la.norm(conn_vec_p, 2) # Length of that vector
					l = conn_vec_l*(1+(w_1-w_2))/2 - r_a[i]

					# Find parameters a, b that describe the separating line (i.e. a'x + b = 0)
					ridge_a = conn_vec_p
					# ridge_b = -conn_vec_p.dot(p[0] + conn_vec_p/2 - r_a[i]*conn_vec_p/conn_vec_l)
					ridge_b = -conn_vec_p.dot(p[0] + l*conn_vec_p/conn_vec_l)
					# Append to inequality matrix and vector
					H_a.append(ridge_a)
					g_a.append(ridge_b)

			H_t[i] = np.array(H_a)
			g_t[i] = np.array(g_a)

	elif n_a == 2:
		deltas = get_agent_distances(A, abs_t, xf_reached, r_a)

		for i in range(n_a):
			conn_vec_p = A[1-i] - A[i]
			conn_vec_l = np.linalg.norm(conn_vec_p, 2)
			ridge_a = conn_vec_p
			# ridge_b = -conn_vec_p.dot(A[i] + conn_vec_p/2 - r_a[i]*conn_vec_p/conn_vec_l)
			ridge_b = -conn_vec_p.dot(A[i] + deltas[i]*conn_vec_p/conn_vec_l)
			H_t[i] = ridge_a.reshape((1, A.shape[1]))
			g_t[i] = ridge_b.reshape((1, 1))

	else:
		H_t = None
		g_t = None

	return H_t, g_t

# Solve linear program to get pair-wise distances between points
def get_agent_distances(A, abs_t, xf_reached, r_a):
	"""
	- A: agent states at abs_t (num_agents, dimension)
	- abs_t: absolute timestep
	- xf_reached: number of timesteps it took for each agent to reach its goal state (num_agents,)
	- r_a: agent radius (num_agens,)
	"""
	n_a = A.shape[0]

	if n_a == 1:
		return None

	d = cp.Variable(n_a)
	pairs = map(list, itertools.combinations(range(n_a), 2))

	# Constraints
	constr = [d >= 0]
	for p in pairs:
		# r_i+r_j <= ||x_i-x_j||_2
		# print('d[%i] + d[%i] <= %g' % (p[0], p[1], la.norm(agent_xcls[p[0]][:2,i]-agent_xcls[p[1]][:2,i], ord=2)-2*r_a))
		constr += [d[p[0]]+d[p[1]] <= la.norm(A[p[0]]-A[p[1]], ord=2)-(r_a[p[0]]+r_a[p[1]])]
		# constr += [d[p[0]]+d[p[1]] <= la.norm(agent_xcls[p[0]][:2,i]-agent_xcls[p[1]][:2,i], ord=np.inf)-2*r_a]

	# Cost
	if n_a == 2:
		w = np.array([min(abs_t+1,xf_reached[j]) for j in range(n_a)])
	else:
		w = np.array([la.norm(A[j]-xf[j][:2], ord=2) for j in range(n_a)])
	cost = w*d

	problem = cp.Problem(cp.Maximize(cost), constr)
	try:
		problem.solve()
	except Exception as e:
		print(e)

	if problem.status == 'infeasible':
		raise(ValueError('Optimization was infeasible for step %i' % abs_t))
	elif problem.status == 'unbounded':
		raise(ValueError('Optimization was unbounded for step %i' % abs_t))

	return d.value

# Get ellipsoidal exploration constraints
def get_traj_ell_con(agent_xcls, xf, r_a=None, tol=-7):
	ball_con = []
	n_a = len(agent_xcls)

	if n_a == 1:
		return None

	n_x = agent_xcls[0].shape[0]
	traj_lens = [agent_xcls[i].shape[1] for i in range(n_a)]
	max_traj_len = np.amax(traj_lens)

	if r_a is None:
		r_a = [0 for _ in range(n_a)]

	# Pad trajectories so they are all the same length
	xf_reached = np.zeros(n_a)
	for i in range(n_a):
		if traj_lens[i] < max_traj_len:
			n_rep = max_traj_len - traj_lens[i]
			agent_xcls[i] = np.append(agent_xcls[i], np.tile(agent_xcls[i][:,-1].reshape((n_x,1)), n_rep), axis=1)
		xcls_norms = la.norm(agent_xcls[i] - xf[i], ord=2, axis=0)
		xf_reached[i] = np.where(xcls_norms <= 10**tol)[0][0]

	# Solve for pair-wise distances between trajectory points for each agent at each time step
	for i in range(max_traj_len):
		# print('Timestep %i' % i)
		A = np.vstack([agent_xcls[j][:2,i] for j in range(n_a)])
		d = get_agent_distances(A, i, xf_reached, r_a)
		ball_con.append(d)

	ball_con = np.array(ball_con).T

	return ball_con

# Get linear exploration constraints using Voronoi graphs
def get_traj_lin_con(agent_xcls, xf, r_a=None, tol=-7):
	n_a = len(agent_xcls)

	if n_a == 1:
		return (None, None)

	H_cl = [[] for _ in range(n_a)]
	g_cl = [[] for _ in range(n_a)]

	n_x = agent_xcls[0].shape[0]
	traj_lens = [agent_xcls[i].shape[1] for i in range(n_a)]
	max_traj_len = np.amax(traj_lens)

	if r_a is None:
		r_a = [0 for _ in range(n_a)]

	# Pad trajectories so they are all the same length
	xf_reached = np.zeros(n_a)
	for i in range(n_a):
		if traj_lens[i] < max_traj_len:
			n_rep = max_traj_len - traj_lens[i]
			agent_xcls[i] = np.append(agent_xcls[i], np.tile(agent_xcls[i][:,-1].reshape((n_x,1)), n_rep), axis=1)
		xcls_norms = la.norm(agent_xcls[i] - xf[i], ord=2, axis=0)
		xf_reached[i] = np.where(xcls_norms <= 10**tol)[0][0]

	# Solve for new set of exploration constraints
	for t in range(max_traj_len):
		A = np.vstack([agent_xcls[j][:2,t] for j in range(n_a)])
		H_t, g_t = get_agent_polytopes(A, t, xf_reached, r_a)
		for i in range(n_a):
			H_cl[i].append(H_t[i])
			g_cl[i].append(g_t[i])
		# pdb.set_trace()

	return zip(H_cl, g_cl)

# A tool for inspecting the trajectory at an iteration, when this function is called, the program will enter into a while loop which waits for user input to inspect the trajectory
def traj_inspector(visualizer, start_t, xcl, x_preds, u_preds, expl_con=None):
	if visualizer is None:
		return
		
	t = start_t

	# Get the max time of the trajectory
	end_times = [xcl_shape[1]-1]
	if expl_con is not None and 'lin' in expl_con:
		end_times.append(len(lin_con[0])-1)
	if expl_con is not None and 'ell' in expl_con:
		end_times.append(len(ball_con)-1)
	max_time = np.amax(end_times)

	print('t = %i' % t)
	print('Press q to exit, f/b to move forward/backwards through iteration time steps')
	while True:
		input = raw_input('(debug) ')
		# Quit inspector
		if input == 'q':
			break
		# Move forward 1 time step
		elif input == 'f':
			if t == max_time:
				print('End reached')
				continue
			else:
				t += 1
				print('t = %i' % t)
				visualizer.plot_state_traj(xcl[:,:min(t,start_t)], x_preds[min(t,start_t-1)], t, expl_con=expl_con, shade=True)
		# Move backward 1 time step
		elif input == 'b':
			if t == 0:
				print('Start reached')
				continue
			else:
				t -= 1
				print('t = %i' % t)
				visualizer.plot_state_traj(xcl[:,:min(t,start_t)], x_preds[min(t,start_t-1)], t, expl_con=expl_con, shade=True)
		else:
			print('Input not recognized')
			print('Press q to exit, f/b to move forward/backwards through iteration time steps')
