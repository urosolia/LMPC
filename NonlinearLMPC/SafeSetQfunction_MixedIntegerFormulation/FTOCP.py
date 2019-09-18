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

	def solve(self, x0, xf):
		# Set initail condition
		x_0_f = x0

		# Set box constraints on states, input and slack
		self.lbx = x0 + [-1000]*(self.n*(self.N)) + [-np.pi/2.0,-1.0]*self.N + [-1000]*self.n + [1]*(self.N-1)
		self.ubx = x0 +  [1000]*(self.n*(self.N)) + [ np.pi/2.0, 1.0]*self.N +  [1000]*self.n + [100000]*(self.N-1)
		
		# Solve nonlinear programm
		self.xGuessTot = np.concatenate((self.xGuess, np.zeros(self.n+self.N-1)), axis=0)
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot.tolist())
		

		# Store optimal solution
		x = np.array(sol["x"])
		self.xSol  = x[0:(self.N+1)*self.n].reshape((self.N+1,self.n)).T
		self.uSol  = x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.N,self.d)).T
		self.slack = x[((self.N+1)*self.n + self.d*self.N):((self.N+1)*self.n + self.d*self.N+self.n)]

		# Check solution flag
		if (self.solver.stats()['success']) and (np.linalg.norm(self.slack,2)< 1e-8):
			self.feasible = 1
		else:
			self.feasible = 0

	def buildNonlinearProgram(self, N, xf):
		# Define variables
		self.N = N
		self.n = 3
		self.d = 2
		n = self.n
		d = self.d
		X     = SX.sym('X', n*(N+1));
		U     = SX.sym('X', d*N);
		slack = SX.sym('X', n);
		slackObs = SX.sym('X', (N-1));

		# Define dynamic constraints
		constraint = []
		for i in range(0, N):
			constraint = vertcat(constraint, X[n*(i+1)+0] - (X[n*i+0] + X[n*i+2]*np.cos(U[d*i+0]))) 
			constraint = vertcat(constraint, X[n*(i+1)+1] - (X[n*i+1] + X[n*i+2]*np.sin(U[d*i+0]))) 
			constraint = vertcat(constraint, X[n*(i+1)+2] - (X[n*i+2] + U[d*i+1])) 


		# Obstacle constraints
		for i in range(1, N):
			constraint = vertcat(constraint, ((X[n*i+0] -  27.0)**2/64.0) + ((X[n*i+1] + 1.0)**2/36.0) - slackObs[i-1])

		constraint = vertcat(constraint, X[n*N:n*(N+1)] - xf + slack) 

		# Defining Cost
		cost = 1000*(slack[0]**2 + slack[1]**2 + slack[2]**2)
		for i in range(0, N):
			cost = cost + 1
			# If needed ma
			# cost = cost + 1e-10*(X[n*i+0]**2 + X[n*i+1]**2 + X[n*i+2]**2 + U[d*i+0]**2 + U[d*i+1]**2 )

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive"}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U,slack, slackObs), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force: 1) n initial constraint, 2) n*N state dynamics and 3) n terminal contraints
		self.lbg_dyanmics = [0]*(n*N) + [0*1.0]*(N-1) + [0]*n
		self.ubg_dyanmics = [0]*(n*N) + [0*100000000]*(N-1)+ [0]*n

	def f(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + x[2]*np.cos(u[0]),
						  x[1] + x[2]*np.sin(u[0]),
						  x[2] + u[1]])
		return xNext.tolist()
