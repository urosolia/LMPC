
import numpy as np
from numpy import linalg as la
import pdb
import copy

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

	def __init__(self, ftocp, l, M):
		self.ftocp = ftocp
		self.SS    = []
		self.uSS   = []
		self.Qfun  = []
		self.l     = l
		self.M     = M
		self.zt    = []
		self.it    = 0

	def addTrajectory(self, x, u):
		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS.append(copy.copy(x))
		self.uSS.append(copy.copy(u))

		# Compute and store the cost associated with the feasible trajectory
		self.Qfun.append(copy.copy(np.arange(x.shape[1]-1,-1,-1)))

		# Update the z vector
		self.zt = x[:, self.ftocp.N]

		# Compute initial guess for nonlinear solver and store few variables
		self.xGuess = np.concatenate((x[:,0:(self.ftocp.N+1)].T.flatten(), u[:,0:(self.ftocp.N)].T.flatten()), axis = 0)
		self.cost    = self.Qfun[-1][0]
		self.oldCost = self.cost + 1
		self.oldIt  = self.it

		# Check if the number of states used to cosntruct the local safe set is greated then the number of closed-loop data
		self.M = np.min([x.shape[1], self.M])

		# Pass inital guess to ftopc object
		self.ftocp.xSol = x[:,0:(self.ftocp.N+1)]
		self.ftocp.uSol = u[:,0:(self.ftocp.N)]
		# Print
		print "Total time added trajectory: ", self.Qfun[-1][0]
		print "Total time old trajectories: ", [self.Qfun[x][0] for x in range(0, self.it)]
		print "self.oldCost", [self.oldCost]
		# Update time Improvement counter
		self.timeImprovement = -1
		self.it = self.it + 1

	def solve(self, xt, verbose = 5):
		# Retrieve variables from ftocp
		N = self.ftocp.N
		n = self.ftocp.n
		d = self.ftocp.d
		
		# Initialize lists which store the solution to the ftocp for different terminal points in the safe set
		inputList = []
		costList  = []
		indList   = []
		xOpenLoop = []
		uOpenLoop = []
		NOpenLoop = []

		# Assign the initial feasible solution 
		self.ftocp.xGuess = self.xGuess

		# Loop for each of the l iterations used to contruct the local safe set
		minIt = np.max([0, self.it - self.l])
		for i in range(minIt, self.it):

			# for iteration l compute the set of indices which are closer to zt
			idx = self.closeToSS(i)

			# Initialize the list which will store the solution to the ftocp for the l-th iteration in the safe set
			costIt  = []
			inputIt = []
			xOpenLoopIt = []
			uOpenLoopIt = []
			NOpenLoopIt = []

			# Now solve the ftocp for each element of the l-th lap which is in the safe set
			totxf = []
			idxToTry = copy.copy(idx)
			for j in idxToTry:
				# Pick terminal point and terminal cost from the safe set
				xf = self.SS[i][:,j]
				Qf = self.Qfun[i][j]
				totxf.append(xf)

				# solve the ftocp (note that verbose = 1 will print just the open loop trajectory and the the output from ipopt)
				verbose_IPOPT = verbose if verbose > 1 else 0
				if self.ftocp.N >= 1: # if the horizon of the ftocp > 1
					if Qf+1 < self.oldCost: # check if the cost of the ftocp could be decreasing, otherwise pointless to solve
						#self.ftocp.xGuess = self.xGuess 

						# Check horizon lenght (basically set manually how many steps you wanna take into SS)
						Ntry = np.min(((self.oldCost-1) - Qf, N)) 
						NOpenLoopIt.append(copy.copy(Ntry))

						# Store old solution and build nonlinear program
						xSolOld = self.ftocp.xSol						
						uSolOld = self.ftocp.uSol					
						self.ftocp.buildNonlinearProgram(Ntry)

						# Solve FTOCP which drive the system from xt to xf
						self.ftocp.solve(xt,xf)

						# Check for feasibility and store the solution
						cost = Ntry + Qf if self.ftocp.feasible else float('Inf')

						# Check if you can extend the trajectory of it is needed to shrink the horizon
						if not self.ftocp.feasible :
							# If not feasible reset horizon and feasible solution
							self.ftocp.xSol = xSolOld
							self.ftocp.uSol = uSolOld
							self.ftocp.buildNonlinearProgram(N)
						else:
							# If feasible cosntruct a solution of lenght N, if possible
							aug_xSol = self.ftocp.xSol.T
							aug_uSol = self.ftocp.uSol.T

							for l in range(j+1, j+1+N-Ntry):
								if l <= (self.SS[i].shape[1]-1):
									aug_xSol = np.vstack((aug_xSol, self.SS[i][:,l].T))
									aug_uSol = np.vstack((aug_uSol, self.uSS[i][:,l-1].T))
									Ntry += 1
									idx[np.where(idxToTry==j)] += 1

							self.ftocp.xSol = aug_xSol.T
							self.ftocp.uSol = aug_uSol.T

							self.ftocp.buildNonlinearProgram(Ntry)

					else: # No need to check if you can drive the system from xt to xf ---> set cost high
						NOpenLoopIt.append(copy.copy(self.ftocp.N))
						cost = 100

				else: # if the horizon of the problem is N = 1 --> the problem is over constraint --> just chack for feasibility
					xNext = self.ftocp.f(xt, self.xGuess[n*(N+1):(n*(N+1)+d)])
					NOpenLoopIt.append(1)
					# check for feasibility and store the solution
					if np.linalg.norm([xNext-xf]) <= 1e-3:
						cost = 1 + Qf 
						self.ftocp.xSol = np.vstack((xt, xf)).T
						self.ftocp.uSol = np.zeros((d,1))
						self.ftocp.uSol[:,0] = self.xGuess[n*(N+1):(n*(N+1)+d)]
					else: 
						cost = float('Inf')
						self.ftocp.xSol = float('Inf') * np.vstack((xt, xf)).T
						self.ftocp.uSol = float('Inf') * np.ones((d,1))
				
				# Store the cost and solution associated with xf. From these solution we will pick and apply the best one
				costIt.append(copy.copy(cost))
				inputIt.append(copy.copy(self.ftocp.uSol[:,0]))

				xOpenLoopIt.append(copy.copy(self.ftocp.xSol))
				uOpenLoopIt.append(copy.copy(self.ftocp.uSol))
				
			# store the solution to the ftocp for the l-th trajectory stored into the local safe set
			indList.append(copy.copy(idx))
			costList.append(copy.copy(costIt))
			inputList.append(copy.copy(inputIt))
			xOpenLoop.append(copy.copy(xOpenLoopIt))
			uOpenLoop.append(copy.copy(uOpenLoopIt))
			NOpenLoop.append(copy.copy(NOpenLoopIt))

		# Pick the best trajectory 
		bestItLocSS, bestTime = np.unravel_index(np.array(costList).argmin(), np.array(costList).shape)	
		bestIt = bestItLocSS + minIt
		# Extract optimal input, open loop and cost 
		self.ut        = inputList[bestItLocSS][bestTime]
		self.xOpenLoop = xOpenLoop[bestItLocSS][bestTime]
		self.uOpenLoop = uOpenLoop[bestItLocSS][bestTime]
		self.cost      =  costList[bestItLocSS][bestTime]

		# Check if the cost is not decreasing (it should not happen). If so apply the open-loop from previous time step
		if  self.oldCost <= self.cost:
			print "The cost is not decreasing"
			self.ut        = self.xGuess[n*(N+1):(n*(N+1)+d)]
			self.xOpenLoop = self.xGuess[0:n*(N+1)].reshape((N+1,n)).T
			self.uOpenLoop = self.xGuess[n*(N+1):].reshape((N,d)).T
			print "Open Loop: ", self.ut 
			print np.round(self.xOpenLoop, decimals =2)
			print np.round(self.uOpenLoop, decimals =2)

			bestItLocSS = self.oldIt
			bestTime    = 0
			bestIt      = bestItLocSS + minIt

			print self.SS[bestIt][:,indList[bestItLocSS][bestTime]]

			self.cost    = self.Qfun[bestIt][indList[bestItLocSS][bestTime]] + N
			self.oldCost = self.cost

			pdb.set_trace()
		

		# Store best it and best cost, and update timeImprovement with respect to previous iteration
		self.oldIt   = bestItLocSS
		self.timeImprovement = self.timeImprovement + self.oldCost - self.cost - 1
		self.oldCost = self.cost

		if verbose>0:
			print "Open Loop Cost:", self.cost, " Time improvement: ",self.timeImprovement
			print np.round(self.xOpenLoop, decimals=2)
			print "Tot Cost List:", costList

			print "Open Loop: ", self.ut 
			print np.round(self.xOpenLoop, decimals =2)
			print np.round(self.uOpenLoop, decimals =2)

		# Now update zt and the horizon length
		# First check if zt == terminal point, if (zt == terminal point) ---> update zt else update horizon N = N - 1
		xflatOpenLoop  = self.xOpenLoop[:,0:(self.ftocp.N+1)].T.flatten() # change format
		uflatOpenLoop  = self.uOpenLoop[:,0:(self.ftocp.N)].T.flatten()
		if ((indList[bestItLocSS][bestTime]+1) <= (self.SS[bestIt].shape[1] - 1)):
			# Update zt taking an extra step into the safe set
			self.zt = self.SS[bestIt][:,indList[bestItLocSS][bestTime]+1]

			# Swift the optimal solution to construct the new candidate solution at the next time step

			self.xGuess[0 : n*N]       = xflatOpenLoop[n:n*(N+1)]
			self.xGuess[n*N : n*(N+1)] = self.zt
			self.xGuess[n*(N+1) : (n*(N+1)+d*(N-1))]     = uflatOpenLoop[d:d*N]
			self.xGuess[(n*(N+1)+d*(N-1)):(n*(N+1)+d*N)] = self.uSS[bestIt][:,indList[bestItLocSS][bestTime]]
		else:
			# Update zt as the x^*_{t+N|t}
			self.zt = xflatOpenLoop[n*N:n*(N+1)]
			if verbose>0:
				print "Changing horizon to ", self.ftocp.N-1

			# Change horizon length
			self.ftocp.buildNonlinearProgram(N = (self.ftocp.N-1))
			# Swift the optimal solution to construct the new candidate solution at the next time step
			nVar = (self.ftocp.N+1)*self.ftocp.n + self.ftocp.N*self.ftocp.d
			self.xGuess = np.zeros(nVar)
 			self.xGuess[0 : n*N]             = xflatOpenLoop[n:n*(N+1)]
			self.xGuess[n*N : (n*N+d*(N-1))] = uflatOpenLoop[d:d*N]

		if np.isinf(self.cost).any():
			print "================== Some of the problem were not feasible. Follows the cost"
			print costList


	def closeToSS(self, it):
		x = self.SS[it]
		u = self.uSS[it]

		oneVec = np.ones((x.shape[1], 1))
		ztVec = (np.dot(oneVec, np.array([self.zt]))).T
		diff = x - ztVec


		norm = la.norm(np.array(diff), 1, axis=0)
		idxMinNorm = np.argsort(norm)

		return idxMinNorm[0:self.M]



		
		