import numpy as np
import scipy as sp

from scipy import sparse, linalg

import osqp

def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    solver = osqp.OSQP()

    if G is not None:
        l = -np.inf * np.ones(len(h))
        if A is not None:
            qp_A = sparse.vstack([G, A]).tocsc()
            qp_l = np.hstack([l, b])
            qp_u = np.hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        solver.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)

    if initvals is not None:
        solver.warm_start(x=initvals)

    res = solver.solve()

    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)

    feasible = False

    if res.info.status_val == osqp.constant('OSQP_SOLVED')\
     or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE')\
     or res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = True

    return res, feasible


def solve_linsys(Q, b, solver):
    if solver == "CVX":
        res_cons = qp(Q, b) # This is ordered as [A B C]
        return np.squeeze(np.array(res_cons['x']))
    elif solver == "OSQP":
        res_cons, _ = osqp_solve_qp(sparse.csr_matrix(Q), b)
        return res_cons.x
    elif solver == "scipy":
        return linalg.solve(Q, -b, sym_pos=True)
    else:
        raise NotImplementedError('only cvx, osqp, scipy supported solvers')
