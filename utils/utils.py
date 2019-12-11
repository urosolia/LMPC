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
	num_ts = 0
	num_iters = len(x_cls)
	n_a = len(x_cls[0])
	n_x = x_cls[0][0].shape[0]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	# Enumerate pairs of agents
	pairs = map(list, itertools.combinations(range(n_a), 2))
	# Get the minimum distance for collision avoidance between agents based on the geometry of their occupied space
	min_dist = []
	for p in pairs:
		if occupied_space is not None:
			if occupied_space['type'] == 'circle':
				dist = occupied_space['params'][p[0]] + occupied_space['params'][p[1]]
			elif occupied_space['type'] == 'box':
				dist = la.norm(occupied_space['params'][p[0]],2) + la.norm(occupied_space['params'][p[1]],2) # Approximate with circle
			else:
				raise(ValueError('Occupied space type not recognized'))
		else:
			dist = 0
		min_dist.append(dist)

	# Get the longest trajectory for all agents
	for iter_cls in x_cls:
		for agent_cl in iter_cls:
			cl_len = agent_cl.shape[1]
			if cl_len > num_ts:
				num_ts = cl_len

	# Set number of time steps to be included to the trajectory length if it was larger
	if num_ts < des_num_ts:
		des_num_ts = num_ts

	if des_num_iters == 'all':
		des_num_iters = num_iters
	if des_num_ts == 'all':
		des_num_ts = num_ts

	# Safe sets in the form of nested lists [[SS_a0_t0, SS_a0_t1, ...], [SS_a1_t0, SS_a1_t1, ...]]
	# SS_a#_t# is a numpy array where the rows correspond to the state dimensions,
	# and the columns correspond to the points in the safe set
	safe_sets = [[] for _ in range(n_a)]
	it_t_ranges = []
	exploration_spaces = [[] for _ in range(n_a)]
	last_invalid_t = -1

	for t in range(num_ts):
		# Determine starting iteration index and ending time step index
		it_start = max(0, num_iters-des_num_iters)
		ts_end = min(num_ts, t+des_num_ts)
		while True:
			# Construct candidate safe set
			print('Constructing safe set from iteration %i to %i and time %i to %i' % (it_start, num_iters-1, t, ts_end-1))
			safe_set_cand_t = [[] for _ in range(n_a)] # Candidate safe sets at this time step
			for a in range(n_a):
				for j in range(it_start, num_iters):
					ss = x_cls[j][a][:,t:ts_end].reshape((n_x,ts_end-t)) # Grab trajectory segments for each iteration included in the safe set
					safe_set_cand_t[a].append(ss)

			# Debug plot
			# plt.figure()
			# for a in range(n_a):
			# 	for j in range(num_iters):
			# 		plt.plot(x_cls[j][a][0,:], x_cls[j][a][1,:], 'o', c=c[a])
			# 	for j in range(it_start, num_iters):
			# 		plt.plot(safe_set_cand_t[a][j][0,:], safe_set_cand_t[a][j][1,:], 'o', c=c[a], markersize=10, markerfacecolor='none')
			# 		for i in range(safe_set_cand_t[a][j].shape[1]):
			# 			plt.plot(safe_set_cand_t[a][j][0,i]+occupied_space['params'][a]*np.cos(np.linspace(0,2*np.pi,100)),
			# 				safe_set_cand_t[a][j][1,i]+occupied_space['params'][a]*np.sin(np.linspace(0,2*np.pi,100)),
			# 				c=c[a])
			# plt.gca().set_aspect('equal')
			# plt.show()

			# Check for potential overlap and minimum distance between agent safe sets
			all_valid = True
			for (p, d) in zip(pairs, min_dist):
				collision = False
				# Collision only defined for position states
				safe_set_pos_0 = np.hstack(safe_set_cand_t[p[0]])
				safe_set_pos_0 = safe_set_pos_0[:2].T
				safe_set_pos_1 = np.hstack(safe_set_cand_t[p[1]])
				safe_set_pos_1 = safe_set_pos_1[:2].T

				# Stack safe set position vectors into data matrix and assign labels
				X = np.append(safe_set_pos_0, safe_set_pos_1, axis=0)
				y = np.append(np.zeros(safe_set_pos_0.shape[0]), np.ones(safe_set_pos_1.shape[0]))

				# Use SVM with linear kernel and no regularization (w'x + b <= 0 for agent p[0])
				clf = svm.SVC(kernel='linear', C=1000)
				clf.fit(X, y)
				w = clf.coef_
				b = clf.intercept_
				# Calculate classifier margin
				margin = 2/la.norm(w, 2)

				# plot_svm_results(X, y, clf)

				# Check for misclassification of support vectors. This indicates that the safe sets are not linearlly separable
				for i in clf.support_:
				    pred_label = clf.predict(X[i].reshape((1,-1)))
				    if pred_label != y[i]:
						collision = True
						print('Potential for collision between agents %i and %i' % (p[0],p[1]))
						break
				# Check for distance between safe sets
				if not collision and margin < d:
					print('Margin between safe sets for agents %i and %i is too small' % (p[0],p[1]))

				# If collision is possible or margin is less than minimum required distance between safe sets, reduce safe set
				# iteration and/or time range
				if collision or margin < d:
					all_valid = False
					it_start += 1
					if it_start >= num_iters:
						it_start = max(0, num_iters-des_num_iters)
						ts_end -= 1

					# Update the time step when a range reduction was last required, we will use this at the end to iterate through
					# the safe sets up to this time and make sure that all safe sets use the same iteration and time range
					last_invalid_t = t
					break

			# all_valid flag is true if all pair-wise collision and margin checks were passed
			if all_valid:
				# Save iteration and time range from this time step, start with these values next time step
				des_num_iters = num_iters - it_start
				des_num_ts = ts_end - t
				it_t_ranges.append({'n_it' : des_num_iters, 'n_t' : des_num_ts})
				print('Safe set construction successful for t = %i, using iteration range %i and time range %i for next time step' % (t, des_num_iters, des_num_ts))
				break # Break from while loop

		for a in range(n_a):
			safe_sets[a].append(safe_set_cand_t[a])
		# pdb.set_trace()

	# Adjust safe sets from before last_invalid_t to have the same iteration and time range
	# and stack trajectory segments from all iterations into a single matrix
	if last_invalid_t >= 0:
		for t in range(last_invalid_t+1):
			for a in range(n_a):
				n_it = it_t_ranges[t]['n_it']
				if n_it > des_num_iters:
					for j in range(n_it-des_num_iters):
						safe_sets[a][t].pop(0)
				n_t = it_t_ranges[t]['n_t']
				if n_t > des_num_ts:
					for j in range(len(safe_sets[a][t])):
						safe_sets[a][t][j] = safe_sets[a][t][j][:,:des_num_ts]
				safe_sets[a][t] = np.hstack(safe_sets[a][t])
	else:
		for t in range(num_ts):
			for a in range(n_a):
				safe_sets[a][t] = np.hstack(safe_sets[a][t])

	pdb.set_trace()

	return safe_sets, exploration_spaces

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
