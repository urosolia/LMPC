
import numpy as np
from numpy import linalg as la
import pdb
import copy

class LMPC(object):
	"""Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, ftocp, CVX):
		# Initialization
		self.ftocp = ftocp
		self.SS    = []
		self.uSS   = []
		self.Qfun  = []
		self.Q = ftocp.Q
		self.R = ftocp.R
		self.it    = 0
		self.CVX = CVX

	def addTrajectory(self, x, u, xf=None):
		if xf is None:
			xf = np.zeros((self.ftocp.n))

		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS.append(copy.copy(x))
		self.uSS.append(copy.copy(u))

		# Compute and store the cost associated with the feasible trajectory
		cost = self.computeCost(x, u, xf)
		self.Qfun.append(cost)

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
		self.it = self.it + 1
		print "Trajectory added to the Safe Set. Current Iteration: ", self.it
		print "Performance stored trajectories: \n", [self.Qfun[i][0] for i in range(self.it)]

	def computeCost(self, x, u, xf):
		l = x.shape[1]
		# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
		for i in range(l):
			idx = l-1-i
			if i == 0: # Terminal cost
				# cost = [np.dot(np.dot(x[:,idx],self.Q),x[:,idx])]
				cost = [self.ftocp.term_cost_fun(x[:,idx], xf).value]
			else:
				# cost.append(np.dot(np.dot(x[:,idx],self.Q),x[:,idx]) + np.dot(np.dot(u[:,idx],self.R),u[:,idx]) + cost[-1])
				cost.append(self.ftocp.stage_cost_fun(x[:,idx], xf, u[:,idx]).value + cost[-1])
		# Finally flip the cost to have correct order
		return np.flip(cost).tolist()

	def solve(self, xt, xf=None, verbose=True):
		# Solve the FTOCP. Here set terminal constraint = ConvHull(self.SS) and terminal cost = BarycentricInterpolation(self.Qfun)
		return self.ftocp.solve(xt, xf, verbose, self.SS, self.Qfun, self.CVX)

	def get_safe_set_q_func(self):
		return (self.SS, self.uSS, self.Qfun)
