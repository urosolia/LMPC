import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy
import datetime
import os
from casadi import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from cvxpy import *

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def main():
	# Check if a storedData folder exist.	
	if not os.path.exists('sampleData'):
		os.makedirs('sampleData')

	option = 0 # Options are: 0 for random or 1 for CS_check
	
	it = 1
	totSamples = 10**5
	P = 15
	totResult = []
	# Loop over iterations
	for it in range(0, 10):
		# Check Assumption 4 via sampling
		numberSuccessful, numberUnSuccessful =runTest(it, totSamples, P, option, showPlots=False)
		print "numberSuccessful", numberSuccessful
		print "numberUnSuccessful", numberUnSuccessful
		result = [numberSuccessful, numberUnSuccessful, totSamples]
		np.savetxt('sampleData/sampleResult_'+str(it)+'_samples_'+str(totSamples)+'.txt', result, fmt='%f' )
		totResult.append(result)

	np.savetxt('sampleData/sampleResultTot_samples_'+str(totSamples)+'.txt', np.array(totResult), fmt='%f' )


###### Test Function #####

def runTest(it, totSamples, Points, option, showPlots=False):
	# Load closed-loop trajectories
	P  = Points
	if it == 0:
		xcl = np.loadtxt('storedData/closedLoopFeasible.txt')
		ucl = np.loadtxt('storedData/inputFeasible.txt')
	else:
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P)+'.txt')
		ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P)+'.txt')
	
	xcl = xcl.T
	ucl = ucl.T
	ucl = np.concatenate((ucl, np.zeros((2,1))) , axis=1)

	# Initializing parameters
	n = xcl.shape[0]
	succTestedPoints = []
	unsuccTestedPoints = []
	numberSuccessful = 0
	withNewWarmStart = 0
	withNewWarmStart1 = 0

	# Pick terminal state or terminal set
	terminalSet = True
	if terminalSet == True:	
		xclFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
		roadHalfWidth = 2.0
		sFinishLine = xclFeasible[0,-1]
		delta_s = 0.5 #2.0
		Xf = np.array([[sFinishLine, -roadHalfWidth, 0.0],[sFinishLine+delta_s, roadHalfWidth, 0.0]]).T
		print "Box Xf"	
		print Xf
		Xf_vertices = np.concatenate((Xf,Xf), axis = 1 )
		Xf_vertices[1,2] = roadHalfWidth
		Xf_vertices[1,3] = -roadHalfWidth
		print "Verices Xf"	
		print Xf_vertices

	# Randomly sample initial conditions
	for k in range(0,totSamples):
		# Pick n+1 points 
		if option == 0:
			points = np.random.choice(xcl.shape[1], n+1, replace=False)
		else:
			idx_time = np.random.choice(xcl.shape[1], 1, replace=False)
			if idx_time[0]+Points < xcl.shape[1]-1:
				points = np.arange(idx_time[0], idx_time[0]+Points,1)
			else:
				points = np.arange(idx_time[0], xcl.shape[1],1)

		X = xcl[:, points]
		U = ucl[:, points]

		# Pick multipliers lambda
		if X.shape[1] > X.shape[0]+1:
			lamb = np.zeros(X.shape[1])
			idxLamb = np.random.choice(X.shape[1], X.shape[0]+1, replace=False)
			dummy = np.random.uniform(0,1, X.shape[0]+1)
			lamb[idxLamb]  = dummy / np.sum(dummy)
		else:
			dummy = np.random.uniform(0,1, X.shape[1])
			lamb  = dummy / np.sum(dummy)

		# Computed the succesor states of the selected points X
		idxNext = points+1
		if (np.max(idxNext > xcl.shape[1]-1)) and (terminalSet):
			idxNext[idxNext>xcl.shape[1]-1] = xcl.shape[1]-1 # 
			X_next = np.concatenate( (xcl[:, idxNext], Xf_vertices), axis = 1)
		else:
			idxNext[idxNext>xcl.shape[1]-1] = xcl.shape[1]-1 # 
			X_next = xcl[:, idxNext]

		# Check if the property holds at the sampled states x = X*lambda
		sampleChecking = sampleCheck(X_next, X, U, lamb)
		sampleChecking.solve(U,lamb)

		# Now check the property was satisfied
		if sampleChecking.feasible == 1:
			X_lamb = np.dot(X,lamb)
			succTestedPoints.append(X_lamb)
			numberSuccessful += 1
			print "Tested points: ",k, ". Unfeasible points: ", len(unsuccTestedPoints), " New warm start: ", withNewWarmStart, " ",withNewWarmStart1
		else:
			# if now try a different warm start
			new_lamb = np.zeros(X.shape[1])/X.shape[1]
			sampleChecking.solve(U,new_lamb)
			if sampleChecking.feasible == 1:
				print "Solved with new warm start"
				X_lamb = np.dot(X,lamb)
				succTestedPoints.append(X_lamb)
				numberSuccessful += 1
				withNewWarmStart += 1
			else:
				sampleChecking.solve(U,0*new_lamb)

				if sampleChecking.feasible == 1:
					print "Solved with new warm start1"
					X_lamb = np.dot(X,lamb)
					succTestedPoints.append(X_lamb)
					numberSuccessful += 1
					withNewWarmStart1 += 1
				else:
					X_lamb = np.dot(X,lamb)
					unsuccTestedPoints.append(X_lamb)
					if showPlots == True:
						plt.figure()
						plt.plot(xcl[0,:], xcl[1,:], 'dk')
						plt.plot(X[0,:], X[1,:], 'ob')
						plt.plot(X_lamb[0], X_lamb[1], 'xr')		

						fig = plt.figure()
						ax = fig.add_subplot(111, projection='3d')
						ax.plot(xcl[0,:], xcl[1,:], xcl[2,:], 'dk')
						ax.plot(X[0,:], X[1,:], X[2,:], 'ob')
						X_lamb = np.dot(X,lamb)
						ax.plot(np.array([X_lamb[0]]), np.array([X_lamb[1]]), np.array([X_lamb[2]]), 'xr')

	# All sampled states have been tested. Now plot the results.
	if numberSuccessful == totSamples:
		print "All points feasible!!!!"
	else:
		print "Feasible Points: ", numberSuccessful
		print "Tested Points: ", totSamples
	
	plt.figure()
	# Plot tested states and closed-loop trajectories in 2D
	arraySuccTestedPoints = np.array(succTestedPoints).T
	plt.plot(arraySuccTestedPoints[0,:], arraySuccTestedPoints[1,:], 'xr', label = 'Tested States')
	plt.plot(xcl[0,:], xcl[1,:], 'ob', label = 'Stored States')	
	plt.xlabel('$s$', fontsize=20)
	plt.ylabel('$e$', fontsize=20)
	plt.legend()
	plt.savefig('sampleData/iteration_'+str(it)+'_2D.png')

	# Plot tested states and closed-loop trajectories in 3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(arraySuccTestedPoints[0,:], arraySuccTestedPoints[1,:], arraySuccTestedPoints[2,:], 'xr', label = 'Tested States')
	ax.plot(xcl[0,:], xcl[1,:], xcl[2,:], 'ob', label = 'Stored States')	
	ax.set_xlabel('$s$', fontsize=20)
	ax.set_ylabel('$e$', fontsize=20)
	ax.set_zlabel('$v$', fontsize=20)
	plt.legend()
	plt.savefig('sampleData/iteration_'+str(it)+'_3D.png')

	if showPlots == True:
		plt.show()

	return numberSuccessful, len(unsuccTestedPoints),
	

# ===================================================================================
# =============================== Defining Object ===================================
# ===================================================================================
class sampleCheck(object):
	def __init__(self, X_next, X, U, lamb):
		# Define variables
		self.dt = 0.5
		self.radius = 10.0
		self.dimLamb = X.shape[1]
		self.buildNonlinearProgram(X_next, X, U, lamb)
		self.feasible = 0


	def solve(self, U, lamb):
		# Set inequality constraints
		# self.lbx = [-10]*self.X.shape[0] + [0]*(self.dimLamb)  + [-np.pi,-1.0]
		# self.ubx =  [10]*self.X.shape[0] + [10]*(self.dimLamb) + [ np.pi, 1.0]
		self.lbx = [0]*(self.dimLamb)  + [-2.0,-1.0]
		self.ubx = [10]*(self.dimLamb) + [ 2.0, 1.0]

		# Compute warm start
		lambda_U  = np.dot(U, lamb)
		# if self.dimLamb == lamb.shape[0]:
		# 	self.xGuessTot = np.concatenate( (np.zeros(self.X.shape[0]), lamb, lambda_U), axis=0 )
		# else:
		# 	self.xGuessTot = np.concatenate( (np.zeros(self.X.shape[0]), np.zeros(self.dimLamb), lambda_U), axis=0 )
		if self.dimLamb == lamb.shape[0]:
			self.xGuessTot = np.concatenate(( lamb, lambda_U), axis=0 )
		else:
			self.xGuessTot = np.concatenate(( np.zeros(self.dimLamb), lambda_U), axis=0 )

		# Solve nonlinear programm
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0=self.xGuessTot.tolist())
				
		# Check solution flag
		# self.slack = sol["x"][0:self.X.shape[0]]
		# if (self.solver.stats()['success']) and (np.linalg.norm(self.slack,2)< 1e-8):
		if (self.solver.stats()['success']):
			self.feasible = 1
			self.solution = sol["x"]

	def buildNonlinearProgram(self, X_next, X, U, lamb):
		self.X_next  = X_next
		self.X       = X
		self.lamb    = lamb
		self.dimLamb = X_next.shape[1]
		
		# Define variables
		gamma   = SX.sym('X',  self.dimLamb)
		U_var   = SX.sym('X', 2)
		# slack   = SX.sym('X', self.X_next.shape[0])

		# X_next * gamma = f( lambda * X, U ) where U = [\theta, a]
		lambda_X   = np.dot(X, lamb)
		constraint = []
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[0,:]) - (lambda_X[0] + self.dt*lambda_X[2]*np.cos( U_var[0] - lambda_X[0] / self.radius) / (1 - lambda_X[1]/self.radius ) )) 
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[1,:]) - (lambda_X[1] + self.dt*lambda_X[2]*np.sin( U_var[0] - lambda_X[0] / self.radius) )) 
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[2,:]) - (lambda_X[2] + self.dt*U_var[1])) 
		# constraint = vertcat(constraint, slack[0] + mtimes(gamma.T, X_next[0,:]) - (lambda_X[0] + self.dt*lambda_X[2]*np.cos( U_var[0] - lambda_X[0] / self.radius) / (1 - lambda_X[1]/self.radius ) )) 
		# constraint = vertcat(constraint, slack[1] + mtimes(gamma.T, X_next[1,:]) - (lambda_X[1] + self.dt*lambda_X[2]*np.sin( U_var[0] - lambda_X[0] / self.radius) )) 
		# constraint = vertcat(constraint, slack[2] + mtimes(gamma.T, X_next[2,:]) - (lambda_X[2] + self.dt*U_var[1])) 
		constraint = vertcat(constraint, mtimes(gamma.T, np.ones((self.dimLamb,1))) - 1) 

		# Defining Cost
		lambda_U   = np.dot(U, lamb)
		cost = (U_var[0]-lambda_U[0])**2 + (U_var[1]-lambda_U[1])**2 #+ 10000000**2*(slack[0]**2 + slack[1]**2 + slack[2]**2)

		# Set IPOPT options
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0} #, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		# nlp = {'x':vertcat(slack, gamma, U_var), 'f':cost, 'g':constraint}
		nlp = {'x':vertcat(gamma, U_var), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force
		self.lbg_dyanmics = [0]*4
		self.ubg_dyanmics = [0]*4

	def solveLinearSystem(self, A, b):
		
		dim = A.shape[1]		
		lamb = Variable(dim)

		constr = [A*lamb == b,
					np.ones(dim)*lamb==1,
					lamb>=0]

		cost = norm(lamb)**2
		
		problem = Problem(Minimize(cost), constr)
		problem.solve(verbose=True)
		

	def f(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + self.dt * x[2]*np.cos(u[0] - x[0]/self.radius) / (1 - x[1] / self.radius),
						  x[1] + self.dt * x[2]*np.sin(u[0] - x[0]/self.radius),
						  x[2] + self.dt * u[1]])
		return xNext.tolist()

if __name__== "__main__":
  main()