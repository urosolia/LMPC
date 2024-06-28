from __future__ import division

import numpy as np
from numpy import linalg as la
import pdb, copy

import utils.utils

class LMPC(object):
	"""Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, ftocp, CVX=False):
		# Initialization
		self.ftocp = ftocp
		# self.SS    = []
		# self.uSS   = []
		self.Qfun  = []
		self.SS_t = []
		self.uSS_t = []
		self.Qfun_t = []

		self.Q = ftocp.Q
		self.R = ftocp.R
		self.it    = 0
		self.CVX = CVX

		self.x_cls = []
		self.u_cls = []

		self.ss_idxs = []

	def addTrajectory(self, x, u, xf=None):
		if xf is None:
			xf = np.zeros((self.ftocp.n))

		n_x = self.ftocp.n
		n_u = self.ftocp.d

		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.x_cls.append(copy.copy(x))
		self.u_cls.append(copy.copy(u))

		# Compute and store the cost associated with the feasible trajectory
		cost = np.array(self.computeCost(x, u, xf))
		self.Qfun.append(cost)

		self.SS_t = []
		self.uSS_t = []
		self.Qfun_t = []
		for t in range(len(self.ss_idxs)-1):
			ss = np.empty((n_x,0))
			uss = np.empty((n_u,0))
			qfun = np.empty(0)
			for (i, j) in enumerate(self.ss_idxs[t]['it_range']):
				ss = np.append(ss, self.x_cls[j][:,self.ss_idxs[t]['ts_range'][i]], axis=1)
				uss = np.append(uss, self.u_cls[j][:,self.ss_idxs[t]['ts_range'][i]], axis=1)
				qfun = np.append(qfun, self.Qfun[j][self.ss_idxs[t]['ts_range'][i]])
			self.SS_t.append(ss)
			self.uSS_t.append(uss)
			self.Qfun_t.append(qfun)

		self.ftocp.costFTOCP = cost[0] + 0.1

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
		self.it = self.it + 1
		print ('Trajectory of length %i added to the Safe Set. Current Iteration: %i' % (x.shape[1], self.it))
		print "Performance of stored trajectories: \n", [self.Qfun[i][0] for i in range(self.it)]

		return cost

	def computeCost(self, x, u, xf):
		l = x.shape[1]
		# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
		for t in range(l-1,-1,-1):
			if t == l-1: # Terminal cost
				# cost = [la.norm((self.Q**0.5).dot(x[:,t]-xf),ord=2)**2]
				# cost = [10*x[1,t]**2]
				cost = [0]
			else:
				if la.norm(x[:,t].reshape((-1,1))-xf,2) <= 1e-5:
					cost.append(0)
				else:
					cost.append(u[:,t].T.dot(self.R).dot(u[:,t]) + 1 + cost[-1])
		# Finally flip the cost to have correct order
		return np.flip(cost).tolist()

	def solve(self, xt, abs_t, xf=None, expl_con=None, verbose=True):
		# Solve the FTOCP at time abs_t. Here set terminal constraint = ConvHull(self.SS) and terminal cost = BarycentricInterpolation(self.Qfun)
		return self.ftocp.solve(xt, abs_t, xf=xf, expl_con=expl_con,
			SS=self.SS_t, Qfun=self.Qfun_t, CVX=self.CVX, verbose=verbose)

	def get_safe_set_q_func(self):
		return (self.SS_t, self.uSS_t, self.Qfun_t)

	def add_safe_set(self, ss_idxs):
		self.ss_idxs = ss_idxs
