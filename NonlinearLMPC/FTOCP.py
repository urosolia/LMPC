from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np

##### MY CODE ######
class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the nonlinear program solved by the above solve methos
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""
	def __init__(self, N):
		# Define variables
		self.N = N
		self.n = 3
		self.d = 2

		self.buildNonlinearProgram(N)


	def solve(self, x0, xf):
		# Set initail condition
		x_0_f = x0

		# Set box constraints on states
		self.lbx = x0 + [-100]*(self.n*(self.N-1))+ xf.tolist() + [-np.pi/2.0,-1.0]*self.N
		self.ubx = x0 +  [100]*(self.n*(self.N-1))+ xf.tolist() + [ np.pi/2.0, 1.0]*self.N

		# Solve nonlinear programm
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		
		# Check solution flag
		if self.solver.stats()['success']:
			self.feasible = 1
		else:
			self.feasible = 0

		# Store optimal solution
		x = np.array(sol["x"])
		self.xSol = x[0:(self.N+1)*self.n].reshape((self.N+1,self.n)).T
		self.uSol = x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.N,self.d)).T

		

	def buildNonlinearProgram(self, N):
		# Define variables
		self.N = N
		self.n = 3
		self.d = 2
		n = self.n
		d = self.d
		X    = SX.sym('X', n*(N+1));
		U    = SX.sym('X', d*N);

		# Define dynamic constraints
		constraint = []
		for i in range(0, N):
			constraint = vertcat(constraint, X[n*(i+1)+0] - (X[n*i+0] + X[n*i+2]*np.cos(U[d*i+0]))) 
			constraint = vertcat(constraint, X[n*(i+1)+1] - (X[n*i+1] + X[n*i+2]*np.sin(U[d*i+0]))) 
			constraint = vertcat(constraint, X[n*(i+1)+2] - (X[n*i+2] + U[d*i+1])) 

		# Obstacle constraints
		for i in range(1, N):
			constraint = vertcat(constraint, ((X[n*i+0] -  27.0)**2/64.0) + ((X[n*i+1] + 1.0)**2/36.0) )

		# Defining Cost
		cost = 0
		for i in range(0, N):
			cost = cost + 1
			# Adding a smal cost to make the Hessian positive definite (Note that IPOPT does it anyway, but with an adaptive strategy)
			cost = cost + 1e-8*(X[n*i+0]**2 + X[n*i+1]**2 + X[n*i+2]**2 + U[d*i+0]**2 + U[d*i+1]**2 )

		# Set IPOPT options
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force: 1) n initial constraint, 2) n*N state dynamics and 3) n terminal contraints
		self.lbg_dyanmics = [0]*(n*N) + [1.0]*(N-1)
		self.ubg_dyanmics = [0]*(n*N) + [1000]*(N-1)

	def f(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + x[2]*np.cos(u[0]),
						  x[1] + x[2]*np.sin(u[0]),
						  x[2] + u[1]])
		return xNext.tolist()
