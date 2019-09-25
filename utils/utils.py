import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.spatial
import cvxpy as cp
import pdb, itertools

def get_agent_polytopes(A, abs_t, traj_lens, r_a):
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

                    v = [] # Vertices that define the ridge
                    p = [point, vor.points[rp[1-rp_idx]]] # Points that the ridge separates
                    for k in range(len(rv)):
                        if rv[k] >= 0:
                            v.append(vor.vertices[rv[k]])

                    conn_vec_p = p[1] - p[0] # Vector from current point to second point
                    conn_vec_l = np.linalg.norm(conn_vec_p, 2) # Length of that vector
                    # Find parameters a, b that describe the separating line (i.e. a'x + b = 0)
                    ridge_a = conn_vec_p #
                    ridge_b = -conn_vec_p.dot(p[0] + conn_vec_p/2 - r_a[i]*conn_vec_p/conn_vec_l)
                    # Append to inequality matrix and vector
                    H_a.append(ridge_a)
                    g_a.append(ridge_b)

            H_t[i] = np.array(H_a)
            g_t[i] = np.array(g_a)

    elif n_a == 2:
        deltas = get_agent_distances(A, abs_t, traj_lens, r_a)

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

def get_agent_distances(A, abs_t, traj_lens, r_a):
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
        w = np.array([min(abs_t+1,traj_lens[j]) for j in range(n_a)])
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
