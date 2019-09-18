
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools


class LMPC(object):
	"""Learning Model Predictive Controller (LMPC)
		Inputs:
			- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
			- l: number of past trajectories used to construct the local safe set and local Q-function
			- M: number of data points from each trajectory used to construct the local safe set and local Q-function 
		Methods:
			- addTrajectory: adds a trajectory to the safe set SS and update value function
			- computeCost: computes the cost associated with a feasible trajectory
			- solve: uses ftocp and the stored data to comptute the predicted trajectory
			- closeToSS: computes the K-nearest neighbors to zt"""

	def __init__(self, ftocp, l, P, verbose):
		self.ftocp = ftocp
		self.SS    = []
		self.uSS   = []
		self.Qfun  = []
		self.l     = l
		self.P     = P
		self.zt    = []
		self.it    = 0
		self.timeVarying = True
		self.itCost = []
		self.verbose = verbose
		self.ftocp.verbose = verbose

	def addTrajectory(self, x, u):
		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS.append(copy.copy(x))
		self.uSS.append(np.concatenate( (copy.copy(u), np.zeros((2,1)) ), axis=1)) # Here concatenating zero as f(xf, 0) = xf by assumption

		# Compute and store the cost associated with the feasible trajectory
		self.Qfun.append(copy.copy(np.arange(x.shape[1]-1,-1,-1)))

		# Update the z vector: this vector will be used to compute the intial guess for the ftocp.
		# Basically zt will be computed using the optimal multipliers lambda and the data stored in the safe set
		zIdx = np.min((self.ftocp.N, np.shape(x)[1]-1))
		self.zt = x[:, zIdx]

		# Compute initial guess for nonlinear solver and store few variables
		self.xGuess = np.concatenate((x[:,0:(self.ftocp.N+1)].T.flatten(), u[:,0:(self.ftocp.N)].T.flatten()), axis = 0)
		self.ftocp.xGuess = self.xGuess

		# Initialize cost varaibles for bookkeeping
		self.cost    = self.Qfun[-1][0]
		self.itCost.append(self.cost)
		self.ftocp.optCost = self.cost + 1
		self.oldIt  = self.it
			
		# Pass inital guess to ftopc object
		self.ftocp.xSol = x[:,0:(self.ftocp.N+1)]
		self.ftocp.uSol = u[:,0:(self.ftocp.N)]
		# Print
		print "Total time added trajectory: ", self.Qfun[-1][0]
		print "Total time stored trajectories: ", [self.Qfun[x][0] for x in range(0, self.it+1)]

		# Update time Improvement counter
		self.timeImprovement = 0

		# Update iteration counter
		self.it = self.it + 1

		# Update indices of stored data points used to contruct the local safe set and Q-function
		self.SSindices =[]
		Tstar = np.min(self.itCost)
		for i in range(0, self.it):
			Tj = np.shape(self.SS[i])[1]-1
			self.SSindices.append(np.arange(Tj - Tstar + self.ftocp.N, Tj - Tstar + self.ftocp.N+self.P))

	def solve(self, xt, verbose = 5):		

		# First retive the data points used to cconstruct the safe set.
		minIt = np.max([0, self.it - self.l])
		SSQfun = []
		SSnext = []
		# Loop over j-l iterations used to contruct the local safe set
		for i in range(minIt, self.it):
			# idx associated with the data points from iteration i which are in the local safe set
			if self.timeVarying == True:
				idx = self.timeSS(i)
			else:
				idx = self.closeToSS(i)
			# Stored state anc cost value
			SSQfun.append( np.concatenate( (self.SS[i][:,idx], [self.Qfun[i][idx]]), axis=0 ).T )

			# Store the successors of the states into the safe set and the control action. 
			# This matrix will be used to compute the vector zt which represent a feasible guess for the ftocp at time t+1
			xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
			extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)
			SSnext.append(extendedSS[:,idx+1].T)


		# From a 3D list to a 2D array
		SSQfun_vector = np.squeeze(list(itertools.chain.from_iterable(SSQfun))).T 
		SSnext_vector = np.squeeze(list(itertools.chain.from_iterable(SSnext))).T 

		# Add dimension if needed
		if SSQfun_vector.ndim == 1:
			SSQfun_vector = np.array([SSQfun_vector]).T
		if SSnext_vector.ndim == 1:
			SSnext_vector = np.array([SSnext_vector]).T

		# Now update ftocp with local safe set
		self.ftocp.buildNonlinearProgram( SSQfun_vector)

		# Now solve ftocp
		self.ftocp.solve(xt, self.zt)			
			
		# Assign input
		self.ut = self.ftocp.uSol[:,0]

		# Update guess for the ftocp using optimal predicted trajectory and multipliers lambda 
		if self.ftocp.optCost > 1:
			xfufNext  = np.dot(SSnext_vector, self.ftocp.lamb)
			# Update zt
			self.zt = xfufNext[0:self.ftocp.n,0]
			# Update initial guess
			xflatOpenLoop  = np.concatenate( (self.ftocp.xSol[:,1:(self.ftocp.N+1)].T.flatten(), xfufNext[0:self.ftocp.n,0]), axis = 0)
			uflatOpenLoop  = np.concatenate( (self.ftocp.uSol[:,1:(self.ftocp.N)].T.flatten()  , xfufNext[self.ftocp.n:(self.ftocp.n+self.ftocp.d),0]), axis = 0)
			self.ftocp.xGuess = np.concatenate((xflatOpenLoop, uflatOpenLoop) , axis = 0)




	def closeToSS(self, it):
		# TO DO: need to add comments. This function is not used in for time-varying, but for space varying.
		x = self.SS[it]
		u = self.uSS[it]

		oneVec = np.ones((x.shape[1], 1))
		ztVec = (np.dot(oneVec, np.array([self.zt]))).T
		diff = x - ztVec


		norm = la.norm(np.array(diff), 1, axis=0)
		idxMinNorm = np.argsort(norm)

		maxIdn = np.min([x.shape[1], self.P])

		return idxMinNorm[0:maxIdn]

	def timeSS(self, it):
		# This function computes the indices used to construct the safe set
		# self.SSindices[it] is initialized when the trajectory is added to the safe set after computing \delta^i and P

		# Read the time indices
		currIdx = self.SSindices[it]
		# By definition we have x_t^j = x_F \forall t > T^j ---> check indices to select
		# currIdxShort = currIdx[ (currIdx >0) & (currIdx < np.shape(self.SS[it])[1])]
		currIdxShort = currIdx[ currIdx < np.shape(self.SS[it])[1] ]
		
		if self.verbose == True:
			print "Time indices selected"
			print currIdxShort

		# Progress time indices
		self.SSindices[it] = self.SSindices[it] + 1

		# If there is just one time index --> add dimension
		if np.shape(currIdxShort)[0] < 1:
			currIdxShort = np.array([np.shape(self.SS[it])[1]-1])

		return currIdxShort


		
		