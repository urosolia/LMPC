from scipy import linalg, sparse
import numpy as np

from datetime import datetime
import pdb

from abc import abstractmethod

from utils.opt_utils import osqp_solve_qp

# solvers.options['show_progress'] = False

class MPC(object):

    def __init__(self, Q, R, P, N, track):
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.track = track

        self.n_x = self.Q.shape[0]
        self.n_u = self.R.shape[0]

        self.x_ref = np.zeros((N+1, self.n_x))
        self.u_ref = np.zeros((N,   self.n_u))

        self.x_pred = np.zeros((N+1, self.n_x))
        self.u_pred = np.zeros((N,   self.n_u))

    def get_preds(self):
        return self.x_pred, self.u_pred

    def update_x_ref(self, x_ref):
        self.x_ref = x_ref

    def update_u_ref(self, u_ref):
        self.u_ref = u_ref

    @abstractmethod
    def solve(self, x):
        raise NotImplementedError("pure virtual")

    @abstractmethod
    def f(self, x, u):
        raise NotImplementedError("pure virtual")

class LinearMPC(MPC):

    from _mpc_opt import (_BuildMatCost, _BuildMatIneqConst, _BuildMatEqConst)

    def __init__(self, A, B, Q, R, P, N, Q_slack, b_slack, R_d, a_lim, df_lim, track):
        super(LinearMPC, self).__init__(Q, R, P, N, track)
        self.A = A
        self.B = B
        self.Q_slack = Q_slack
        self.b_slack = b_slack
        self.R_d = R_d

        self.a_lim = a_lim
        self.df_lim = df_lim

        self.solve_time = 0
        self.feasible = True

        self.M = None
        self.q = None

        self.F = None
        self.b = None

        self.G = None
        self.E = None

    def solve(self, x):
        if self.M is None or self.F is None or self.G is None:
            raise ValueError('QP ingredients have not been constructed!')

        start_time = datetime.now()
#        sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0))
        res_cons, self.feasible = osqp_solve_qp(sparse.csr_matrix(self.M),
            self.q,
            sparse.csr_matrix(self.F),
            self.b,
            sparse.csr_matrix(self.G),
            np.dot(self.E, x))
        end_time = datetime.now()

        self.solve_time = (end_time - start_time).total_seconds()

        sol = res_cons.x

        self.x_pred = np.squeeze(np.transpose(np.reshape(sol[np.arange(self.n_x*(self.N+1))], (self.N+1, self.n_x)))).T
        self.u_pred = np.squeeze(np.transpose(np.reshape(sol[self.n_x*(self.N+1) + np.arange(self.n_u*self.N)], (self.N, self.n_u)))).T

        return self.u_pred, self.x_pred, self.feasible, self.solve_time

    def build_qp(self):
        self.M, self.q = self._BuildMatCost()
        self.F, self.b = self._BuildMatIneqConst()
        self.G, self.E = self._BuildMatEqConst()

    def update_model(self, As, Bs):
        self.A = As
        self.B = Bs
        self.G, self.E = self._BuildMatEqConst()

    def f(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
