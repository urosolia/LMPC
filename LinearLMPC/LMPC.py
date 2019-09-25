
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools

class LMPC(object):
	"""Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, ftocp, CVX, l = 1, M = 4):
		# Initialization
		self.ftocp = ftocp
		self.SS    = []
		self.uSS   = []
		self.Qfun  = []
		self.Q = ftocp.Q
		self.R = ftocp.R
		self.it    = 0
		self.CVX = CVX
		self.l = l
		self.M = M
		self.localSS = False

	def addTrajectory(self, x, u):
		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS.append(copy.copy(x))
		self.uSS.append(copy.copy(u))

		# Compute and store the cost associated with the feasible trajectory
		cost = self.computeCost(x, u)
		self.Qfun.append(cost)

		# Initialize zVector
		self.zt = np.array(x[self.ftocp.N])

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
		self.it = self.it + 1
		print "Trajectory added to the Safe Set. Current Iteration: ", self.it
		print "Performance stored trajectories: \n", [self.Qfun[i][0] for i in range(0, self.it)]

	def computeCost(self, x, u):
		# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
		for i in range(0,len(x)):
			idx = len(x)-1 - i
			if i == 0:
				cost = [np.dot(np.dot(x[idx],self.Q),x[idx])]
			else:
				cost.append(np.dot(np.dot(x[idx],self.Q),x[idx]) + np.dot(np.dot(u[idx],self.R),u[idx]) + cost[-1])
		
		# Finally flip the cost to have correct order
		return np.flip(cost).tolist()

	def solve(self, xt, verbose = False):

		# Solve the FTOCP. 
		# Here set terminal constraint = ConvHull(self.SS) and terminal cost = BarycentricInterpolation(self.Qfun)
		if self.localSS ==False:
			SS_vector = np.squeeze(list(itertools.chain.from_iterable(self.SS))).T # From a 3D list to a 2D array
			Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(self.Qfun))), 0) # From a 2D list to a 1D array
			
			self.ftocp.solve(xt, verbose, SS_vector, Qfun_vector, self.CVX)

		else:
			# Compute Local SS
			# Loop for each of the l iterations used to contruct the local safe set
			minIt = np.max([0, self.it - self.l])
			SSQfun = []
			SSnext = []
			for i in range(minIt, self.it):
				# for iteration l compute the set of indices which are closer to zt
				idx = self.closeToSS(i)
				SSQfun.append( np.concatenate( (self.SS[i][:,idx], [self.Qfun[i][idx]]), axis=0 ).T )

				# Create next SS
				xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
				extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)
				SSnext.append(extendedSS[:,idx+1].T)

			SSQfun_vector = np.squeeze(list(itertools.chain.from_iterable(SSQfun))).T # From a 3D list to a 2D array
			SSnext_vector = np.squeeze(list(itertools.chain.from_iterable(SSnext))).T # From a 3D list to a 2D array
			
			# Solve with local SS
			self.ftocp.solve(xt, verbose, SSQfun_vector[0:self.ftocp.n, :],  SSQfun_vector[self.ftocp.n, :], self.CVX)

			xfufNext  = np.dot(SSnext_vector, self.ftocp.lamb)

			xflatOpenLoop  = np.concatenate( (self.ftocp.xSol[:,1:(self.ftocp.N+1)].T.flatten(), xfufNext[0:self.ftocp.n,0]), axis = 0)
			uflatOpenLoop  = np.concatenate( (self.ftocp.uSol[:,1:(self.ftocp.N)].T.flatten()  , xfufNext[self.ftocp.n:(self.ftocp.n+self.ftocp.d),0]), axis = 0)
			self.ftocp.xGuess = np.concatenate((xflatOpenLoop, uflatOpenLoop) , axis = 0)
			self.zt = xfufNext[0:self.ftocp.n,0]


		# Update predicted trajectory
		self.xPred= self.ftocp.xPred
		self.uPred= self.ftocp.uPred


	def closeToSS(self, it):
		x = self.SS[it]
		u = self.uSS[it]

		xtr = np.array(x).T
		utr = np.array(u).T

		oneVec = np.ones((xtr.shape[1], 1))
		ztVec = (np.dot(oneVec, np.array([self.zt]))).T
		diff = xtr - ztVec


		norm = la.norm(np.array(diff), 1, axis=0)
		idxMinNorm = np.argsort(norm)

		return idxMinNorm[0:self.M]		