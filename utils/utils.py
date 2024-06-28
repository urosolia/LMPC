from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.spatial
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import svm
import pdb, itertools, matplotlib

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

def get_safe_set(x_cls, xf, des_num_ts='all', des_num_iters='all', occupied_space=None):
	n_a = len(x_cls[0])
	n_x = x_cls[0][0].shape[0]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	# Enumerate pairs of agents
	pairs = map(list, itertools.combinations(range(n_a), 2))
	# Get the minimum distance for collision avoidance between agents based on the geometry of their occupied space
	min_dist = []
	r_a = np.zeros(n_a)
	for p in pairs:
		if occupied_space is not None:
			if occupied_space['type'] == 'circle':
				dist = occupied_space['params'][p[0]] + occupied_space['params'][p[1]]
				r_a[p[0]] = occupied_space['params'][p[0]]
				r_a[p[1]] = occupied_space['params'][p[1]]
			elif occupied_space['type'] == 'box':
				dist = la.norm(occupied_space['params'][p[0]],2) + la.norm(occupied_space['params'][p[1]],2) # Approximate with circle
				r_a[p[0]] = la.norm(occupied_space['params'][p[0]],2)
				r_a[p[1]] = la.norm(occupied_space['params'][p[1]],2)
			else:
				raise(ValueError('Occupied space type not recognized'))
		else:
			dist = 0
		min_dist.append(dist)

	num_ts = 0
	num_iters = len(x_cls)
	cl_lens = []

	# Get the longest trajectory over iterations of interest for all agents
	it_start = max(0, num_iters-des_num_iters)
	orig_range = range(it_start, num_iters)
	for j in orig_range:
		iter_cls = x_cls[j]
		it_cl_lens = []
		for agent_cl in iter_cls:
			it_cl_lens.append(agent_cl.shape[1])
			if agent_cl.shape[1] > num_ts:
				num_ts = agent_cl.shape[1]
		cl_lens.append(it_cl_lens)

	# pdb.set_trace()

	# Set number of time steps to be included to the trajectory length if it was larger
	if num_ts < des_num_ts:
		des_num_ts = num_ts

	if des_num_iters == 'all':
		des_num_iters = num_iters
	if des_num_ts == 'all':
		des_num_ts = num_ts

	# safe_set_idxs = [agent_0_ss_idxs, agent_1_ss_idxs, ... , agent_M_ss_idxs]
	# agent_#_ss_idxs = [ss_idxs_0, ss_idxs_1, ... , ss_idxs_T]
	safe_sets_idxs = [[] for _ in range(n_a)]
	exploration_spaces = [[[], []] for _ in range(n_a)]
	last_invalid_t = -1

	for t in range(num_ts):
		# Determine starting iteration index and ending time step index
		it_start = max(0, num_iters-des_num_iters)
		# ts_end = min(num_ts, t+des_num_ts)
		ts_end = t+des_num_ts
		H_t = [[] for _ in range(n_a)]
		g_t = [[] for _ in range(n_a)]
		while True:
			# Construct candidate safe set
			print('Constructing safe set from iteration %i to %i and time %i to %i' % (it_start, num_iters-1, t, ts_end-1))
			safe_set_cand_t = []
			for a in range(n_a):
				it_range = range(it_start, num_iters)
				ts_range = []
				for j in it_range:
					i = orig_range.index(j)
					ts_range.append(range(min(t, cl_lens[i][a]-1), min(ts_end, cl_lens[i][a])))
					print(range(min(t, cl_lens[i][a]-1), min(ts_end, cl_lens[i][a])), x_cls[j][a].shape)
				ss_idxs = {'it_range' : it_range, 'ts_range' : ts_range}
				safe_set_cand_t.append(ss_idxs) # Candidate safe sets at this time step

			# Check for potential overlap and minimum distance between agent safe sets
			all_valid = True
			for (p, d) in zip(pairs, min_dist):
				collision = False
				# Collision only defined for position states
				safe_set_pos_0 = np.empty((2,0))
				safe_set_pos_1 = np.empty((2,0))
				for (i, j) in enumerate(safe_set_cand_t[p[0]]['it_range']):
					safe_set_pos_0 = np.append(safe_set_pos_0, x_cls[j][p[0]][:2,safe_set_cand_t[p[0]]['ts_range'][i]], axis=1)
					safe_set_pos_1 = np.append(safe_set_pos_1, x_cls[j][p[1]][:2,safe_set_cand_t[p[1]]['ts_range'][i]], axis=1)

				# Stack safe set position vectors into data matrix and assign labels agent p[0]: -1, agent p[1]: 1
				X = np.append(safe_set_pos_0, safe_set_pos_1, axis=1).T
				y = np.append(-np.ones(safe_set_pos_0.shape[1]), np.ones(safe_set_pos_1.shape[1]))

				# if t == 68:
				# 	pdb.set_trace()

				# Use SVM with linear kernel and no regularization (w'x + b <= -a_0 for agent p[0], w'x + b >= a_1 for agent p[1])
				clf = svm.SVC(kernel='linear', C=1000)
				clf.fit(X, y)
				w = np.squeeze(clf.coef_)
				b = np.squeeze(clf.intercept_)

				# Calculate classifier margin
				margin = 2/la.norm(w, 2)

				# Check for misclassification of support vectors. This indicates that the safe sets are not linearlly separable
				for i in clf.support_:
					pred_label = clf.predict(X[i].reshape((1,-1)))
					# pred_val = clf.decision_function(X[i].reshape((1,-1)))
				if pred_label != y[i]:
					collision = True
					print('Potential for collision between agents %i and %i' % (p[0],p[1]))
					break
				# Check for distance between safe sets
				if not collision and margin < d:
					print('Margin between safe sets for agents %i and %i is too small' % (p[0],p[1]))

				# If collision is possible or margin is less than minimum required distance between safe sets, reduce safe set
				# iteration and/or time range
				# Currently, we reduce iteration range first. If iteration range cannot be reduced any further then we reduce time step range
				if collision or margin < d:
					all_valid = False
					it_start += 1
					if it_start >= num_iters:
						it_start = max(0, num_iters-des_num_iters)
						ts_end -= 1

					# Update the time step when a range reduction was last required, we will use this at the end to iterate through
					# the safe sets up to this time and make sure that all safe sets use the same iteration and time range
					last_invalid_t = t

					# Reset the candidate exploration spaces
					H_t = [[] for _ in range(n_a)]
					g_t = [[] for _ in range(n_a)]
					break

				# Distance between hyperplanes is (a_0+a_1)/\|w\|
				a_0_min = d*la.norm(w, 2)/(1 + r_a[p[1]]/r_a[p[0]])
				a_1_min = d*la.norm(w, 2)/(1 + r_a[p[0]]/r_a[p[1]])

				ratio_remain_0 = la.norm(x_cls[0][p[0]][:2,-1] - safe_set_pos_0[:,0], 2)/la.norm(x_cls[0][p[0]][:2,-1] - x_cls[0][p[0]][:2,0], 2)
				ratio_remain_1 = la.norm(x_cls[0][p[1]][:2,-1] - safe_set_pos_1[:,0], 2)/la.norm(x_cls[0][p[1]][:2,-1] - x_cls[0][p[1]][:2,0], 2)
				w_0 = np.exp(35*ratio_remain_0-3)/(np.exp(35*ratio_remain_0-3)+1)
				w_1 = np.exp(35*ratio_remain_1-3)/(np.exp(35*ratio_remain_1-3)+1)

				# Solve for tight hyperplane bounds for both collections of points
				z = cp.Variable(1)
				cost = z
				constr = []
				for i in range(safe_set_pos_0.shape[1]):
					constr += [w.dot(safe_set_pos_0[:,i]) + b <= z]
				problem = cp.Problem(cp.Minimize(cost), constr)
				problem.solve(solver=cp.MOSEK, verbose=False)
				a_0_max = -z.value[0]

				z = cp.Variable(1)
				cost = z
				constr = []
				for i in range(safe_set_pos_1.shape[1]):
					constr += [-w.dot(safe_set_pos_1[:,i]) - b <= z]
				problem = cp.Problem(cp.Minimize(cost), constr)
				problem.solve(solver=cp.MOSEK, verbose=False)
				a_1_max = -z.value[0]

				if a_0_max > a_0_min and a_1_max > a_1_min:
					if w_0 <= w_1:
						a_shift = (a_0_max - a_0_min)*(1-w_0/w_1)
						a_0 = a_0_min + a_shift
						a_1 = a_1_min - a_shift
					else:
						a_shift = (a_1_max - a_1_min)*(1-w_1/w_0)
						a_0 = a_0_min - a_shift
						a_1 = a_1_min + a_shift
				else:
					a_0 = a_0_max - 1e-5 # Deal with precision issues when a point in the safe set is on the exploration space boundary
					a_1 = a_1_max - 1e-5

				# print('-------------------------')
				# print(p)
				# print(margin, d)
				# print(w_0, a_0_min, a_0, a_0_max)
				# print(w_1, a_1_min, a_1, a_1_max)
				# if (t == 17):
				# 	pdb.set_trace()

				# Exploration spaces
				H_t[p[0]].append(w)
				g_t[p[0]].append(b+a_0)
				H_t[p[1]].append(-w)
				g_t[p[1]].append(-b+a_1)

				# plot_svm_results(X, y, clf)

			# all_valid flag is true if all pair-wise collision and margin checks were passed
			if all_valid:
				# Save iteration and time range from this time step, start with these values next time step
				des_num_iters = num_iters - it_start
				des_num_ts = ts_end - t
				for a in range(n_a):
					H_t[a] = np.array(H_t[a])
					g_t[a] = np.array(g_t[a])
				print('Safe set construction successful for t = %i, using iteration range %i and time range %i for next time step' % (t, des_num_iters, des_num_ts))
				break # Break from while loop

		# Debug plot
		# plt.figure()
		# ax = plt.gca()
		#
		# for a in range(n_a):
		# 	ss_cand = safe_set_cand_t[a]
		# 	for j in range(num_iters):
		# 		plt.plot(x_cls[j][a][0,:], x_cls[j][a][1,:], 'o', c=c[a])
		# 	for (i, j) in enumerate(ss_cand['it_range']):
		# 		for i in ss_cand['ts_range'][i]:
		# 			plt.plot(x_cls[j][a][0,i]+occupied_space['params'][a]*np.cos(np.linspace(0,2*np.pi,100)),
		# 				x_cls[j][a][1,i]+occupied_space['params'][a]*np.sin(np.linspace(0,2*np.pi,100)),
		# 				c=c[a])
		#
		# 	xlim = np.array([-2.5, 2.5])
		# 	ylim = [-1.5, 1.5]
		# 	# xx = np.linspace(xlim[0], xlim[1], 30)
		# 	# yy = np.linspace(ylim[0], ylim[1], 30)
		# 	# YY, XX = np.meshgrid(yy, xx)
		# 	# xy = np.vstack([XX.ravel(), YY.ravel()])
		# 	#
		# 	# z = (H_t[a].dot(xy) + g_t[a].reshape((-1,1)) <= 0)
		# 	# z = z[0] & z[1]
		# 	# true_idx = np.where(z)
		# 	# plt.scatter(xy[0,true_idx], xy[1,true_idx], facecolors='none', c=c[a], s=5)
		#
		# 	y_0 = (-H_t[a][0,0]*xlim-g_t[a][0])/H_t[a][0,1]
		# 	y_1 = (-H_t[a][1,0]*xlim-g_t[a][1])/H_t[a][1,1]
		# 	plt.plot(xlim, y_0, c=c[a])
		# 	plt.plot(xlim, y_1, c=c[a])
		#
		# ax.set_xlim(xlim)
		# ax.set_ylim(ylim)
		# ax.set_aspect('equal')
		# plt.show()

		# pdb.set_trace()

		for a in range(n_a):
			safe_sets_idxs[a].append(safe_set_cand_t[a])
			exploration_spaces[a][0].append(H_t[a])
			exploration_spaces[a][1].append(g_t[a])

	# Adjust safe sets from before last_invalid_t to have the same iteration and time range and test that safe sets are contained
	# in the exploration spaces at each time step
	for t in range(num_ts-1):
		for a in range(n_a):
			if t <=  last_invalid_t:
				old_it_len = len(safe_sets_idxs[a][t]['it_range'])
				safe_sets_idxs[a][t]['it_range'] = safe_sets_idxs[a][last_invalid_t+1]['it_range'] # Update iteration range
				new_it_len = len(safe_sets_idxs[a][t]['it_range'])
				for _ in range(old_it_len - new_it_len):
					safe_sets_idxs[a][t]['ts_range'].pop(0) # Throw away iterations that we don't include anymore
				for i in range(new_it_len):
					n_ss = len(safe_sets_idxs[a][t]['ts_range'][i])
					if n_ss > des_num_ts:
						safe_sets_idxs[a][t]['ts_range'][i] = safe_sets_idxs[a][t]['ts_range'][i][:des_num_ts] # Update time range for remaining iterations

			safe_set_pos = np.empty((2,0))
			for (i, j) in enumerate(safe_sets_idxs[a][t]['it_range']):
				safe_set_pos = np.append(safe_set_pos, x_cls[j][a][:2,safe_sets_idxs[a][t]['ts_range'][i]], axis=1)
			in_exp_space = (exploration_spaces[a][0][t].dot(safe_set_pos) + exploration_spaces[a][1][t].reshape((-1,1)) <= 0)
			if not np.all(in_exp_space):
				raise(ValueError('Safe set not contained in exploration space at time %i' % t))

	pdb.set_trace()
	return safe_sets_idxs, exploration_spaces

def plot_svm_results(X, y, clf):
	plt.figure()
	plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# create grid to evaluate model
	xx = np.linspace(xlim[0], xlim[1], 30)
	yy = np.linspace(ylim[0], ylim[1], 30)
	YY, XX = np.meshgrid(yy, xx)
	xy = np.vstack([XX.ravel(), YY.ravel()]).T
	Z = clf.decision_function(xy).reshape(XX.shape)

	# plot decision boundary and margins
	ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
			   linestyles=['--', '-', '--'])

	ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
			   linewidth=1, facecolors='none', edgecolors='k')
	plt.show()
