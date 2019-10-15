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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

class sampleCheck(object):
	def __init__(self, X_next, X, U, lamb):
		# Define variables
		self.dt = 0.5
		self.radius = 10.0
		self.buildNonlinearProgram(X_next, X, U, lamb)
		self.feasible = 0


	def solve(self, U, lamb):
		X = self.X

		self.lbx = [0]*(X.shape[1]) + [-np.pi,-1.0]
		self.ubx = [1]*(X.shape[1]) + [ np.pi, 1.0]

		lambda_U  = np.dot(U, lamb)

		self.xGuessTot = np.concatenate( (lamb, lambda_U), axis=0 )

		# Solve nonlinear programm
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0=self.xGuessTot.tolist())
				
		# Check solution flag
		if self.solver.stats()['success']:
			self.feasible = 1
			self.solution = sol["x"]

	def buildNonlinearProgram(self, X_next, X, U, lamb):
		self.X_next = X_next
		self.X      = X
		self.lamb   = lamb
		
		# Define variables
		gamma   = SX.sym('X',  X.shape[1])
		U_var   = SX.sym('X', 2)

		# X_next * gamma = f( lambda * X, U ) where U = [\theta, a]
		lambda_X   = np.dot(X, lamb)
		constraint = []
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[0,:]) - (lambda_X[0] + self.dt*lambda_X[2]*np.cos( U_var[0] - lambda_X[0] / self.radius) / (1 - lambda_X[1]/self.radius ) )) 
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[1,:]) - (lambda_X[1] + self.dt*lambda_X[2]*np.sin( U_var[0] - lambda_X[0] / self.radius) )) 
		constraint = vertcat(constraint, mtimes(gamma.T, X_next[2,:]) - (lambda_X[2] + self.dt*U_var[1])) 
		constraint = vertcat(constraint, mtimes(gamma.T, np.ones((X.shape[1],1))) - 1) 

		# Defining Cost
		lambda_U   = np.dot(U, lamb)
		cost = (U_var[0]-lambda_U[0])**2 + (U_var[1]-lambda_U[1])**2

		# Set IPOPT options
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(gamma,U_var), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force
		self.lbg_dyanmics = [0]*4
		self.ubg_dyanmics = [0]*4

def main():
	P  = 12
	it = 9
	# xcl = np.loadtxt('storedData/closedLoopFeasible.txt')
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	# ucl = np.loadtxt('storedData/inputFeasible.txt')
	ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P)+'.txt')
	ucl = ucl.T
	ucl = np.concatenate((ucl, np.zeros((2,1))) , axis=1)

	n = xcl.shape[0]
	Points = n + 1# (n+1) + points
	succTestedPoints = []
	unsuccTestedPoints = []

	# for k in range(0,1000):
	for k in range(0,100000):
		points = np.random.choice(xcl.shape[1], Points, replace=False)

		dummy = np.random.uniform(0,1, Points)
		lamb  = dummy / np.sum(dummy)

		X = xcl[:, points]
		U = ucl[:, points]
		idxNext = points+1

		idxNext[idxNext>xcl.shape[1]-1] = xcl.shape[1]-1
		X_next = xcl[:, idxNext]

		sampleChecking = sampleCheck(X_next, X, U, lamb)
		sampleChecking.solve(U,lamb)

		if sampleChecking.feasible == 1:
			X_lamb = np.dot(X,lamb)
			succTestedPoints.append(X_lamb)
			print "feasible point: ",k
		else:
			sampleChecking.solve(U,lamb*0)
			if sampleChecking.feasible == 1:
				print "Solved with new warm start"
				X_lamb = np.dot(X,lamb)
				succTestedPoints.append(X_lamb)
			else:	
				plt.figure()
				plt.plot(xcl[0,:], xcl[1,:], 'dk')
				plt.plot(X[0,:], X[1,:], 'ob')
				X_lamb = np.dot(X,lamb)
				unsuccTestedPoints.append(X_lamb)
				plt.plot(X_lamb[0], X_lamb[1], 'xr')		
				print "Not feasible"
				print X
				print X_next

				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.plot(xcl[0,:], xcl[1,:], xcl[2,:], 'dk')
				ax.plot(X[0,:], X[1,:], X[2,:], 'ob')
				X_lamb = np.dot(X,lamb)
				unsuccTestedPoints.append(X_lamb)
				ax.plot(np.array([X_lamb[0]]), np.array([X_lamb[1]]), np.array([X_lamb[2]]), 'xr')

				plt.show()


	plt.figure()
	if unsuccTestedPoints == []:
		print "All points feasible!!!!"

	arraySuccTestedPoints = np.array(succTestedPoints).T
	plt.plot(arraySuccTestedPoints[0,:], arraySuccTestedPoints[1,:], 'xr', label = 'Tested States')
	plt.plot(xcl[0,:], xcl[1,:], 'ob', label = 'Stored States')	
	plt.xlabel('$s$', fontsize=20)
	plt.ylabel('$e$', fontsize=20)
	plt.legend()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(arraySuccTestedPoints[0,:], arraySuccTestedPoints[1,:], arraySuccTestedPoints[2,:], 'xr', label = 'Tested States')
	ax.plot(xcl[0,:], xcl[1,:], xcl[2,:], 'ob', label = 'Stored States')	
	ax.set_xlabel('$s$', fontsize=20)
	ax.set_ylabel('$e$', fontsize=20)
	ax.set_zlabel('$v$', fontsize=20)
	plt.legend()
	plt.show()

if __name__== "__main__":
  main()