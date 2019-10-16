from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *

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
		self.n = 2
		self.d = 1
		self.optCost= np.inf
		self.dt = 0.5#0.25
		self.maxV = 4.0

	def  set_xf(self, xf):
		self.xf = xf

	def solve(self, x0, zt):
		# Initialize initial guess for lambda
		lambGuess = np.ones(self.dimSS)/self.dimSS
		self.xGuessTot = np.concatenate( (self.xGuess, lambGuess), axis=0 )

		# Need to solve N+1 ftocp as the stage cost is the indicator function --> try all configuration
		costSolved = []
		soluSolved = []
		for i in range(0, self.N+1):
			if i is not self.N:
				# Set box constraints on states (here we constraint the last i steps of the horizon to be xf)
				self.lbx = x0 + [-10.0, -0.0]*(self.N-i)+ self.xf.tolist()*i + [-1.0]*self.N + [0]*self.dimSS  
				self.ubx = x0 +  [10.0,  self.maxV]*(self.N-i)+ self.xf.tolist()*i + [ 1.0]*self.N + [10]*self.dimSS

				# Solve nonlinear programm
				sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot.tolist())
			
				# Check if the solution is feasible
				if (self.solver.stats()['success']):
					self.feasible = 1
					# Notice that the cost is given by the cost of the ftocp + the number of steps not constrainted to be xf
					lamb = sol["x"][((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)]
					terminalCost = np.dot(self.costSS, lamb)
					if i == 0:
						costSolved.append(terminalCost+(self.N))
					else:
						costSolved.append(terminalCost+(self.N-(i-1)))
					soluSolved.append(sol)
				else:
					costSolved.append(np.inf)
					soluSolved.append(sol)
					self.feasible = 0

			else: # if horizon one time step just check feasibility of the initial guess
				uGuess = self.xGuess[(self.n*(self.N+1)):(self.n*(self.N+1)+self.d)]
				xNext  = self.f(x0, uGuess)
				if np.dot(xNext-self.xf, xNext-self.xf)<= 1e-8:
					self.feasible = 1
					costSolved.append(1)
					sol["x"] = self.xGuessTot
					soluSolved.append(sol)
				else:
					costSolved.append(np.inf)
					soluSolved.append(sol)
					self.feasible = 0


		# Check if LMPC cost is decreasing (it should as it is a Lyap function)
		if np.min(costSolved) >  self.optCost:
			print "Cost not decreasing: ", self.optCost, np.min(costSolved)
			pdb.set_trace()


		# Store optimal solution
		self.optCost = np.min(costSolved)
		x = np.array(soluSolved[np.argmin(costSolved)]["x"])
		self.xSol = x[0:(self.N+1)*self.n].reshape((self.N+1,self.n)).T
		self.uSol = x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.N,self.d)).T
		self.lamb = x[((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)]

		if self.verbose == True:
			print "Cost Vector:", costSolved, np.argmin(costSolved)
			print "Optimal Solution:", self.xSol

	def buildNonlinearProgram(self, SSQfun):
		# Define variables
		n = self.n
		d = self.d
		N = self.N
		X      = SX.sym('X', n*(N+1));
		U      = SX.sym('X', d*N);
		dimSS  = np.shape(SSQfun)[1]
		lamb   = SX.sym('X',  dimSS)
		xSS    = SSQfun[0:n, :]
		costSS = np.array([SSQfun[-1, :]])
		# slack  = SX.sym('X', n);

		self.SSQfun = SSQfun
		self.xSS = xSS
		self.costSS = costSS
		self.dimSS = dimSS
		self.xSS = xSS
		self.costSS = costSS

		# Define dynamic constraints
		constraint = []
		for i in range(0, N):
			constraint = vertcat(constraint, X[n*(i+1)+0] - (X[n*i+0] + self.dt*X[n*i+1] ) )
			constraint = vertcat(constraint, X[n*(i+1)+1] - (X[n*i+1] + self.dt*(1 - (X[n*i+1] / self.maxV )**2 ) * U[d*i+0] ) )

		# terminal constraints
		constraint = vertcat(constraint, X[n*N:(n*(N+1))+0] - mtimes( xSS ,lamb) )
		constraint = vertcat(constraint, 1 - mtimes(np.ones((1, dimSS )), lamb ) )

		# Defining Cost (We will add stage cost later)
		cost = mtimes(costSS, lamb)

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive"}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U,lamb), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force: 1) n*N state dynamics, 2) n terminal contraints and 3) CVX hull constraint
		self.lbg_dyanmics = [0]*(n*N) + [0]*(n) + [0]
		self.ubg_dyanmics = [0]*(n*N) + [0]*(n) + [0]

	def f(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + self.dt * x[1],
						  x[1] + self.dt * (1 - (x[1] / self.maxV)**2 ) * u[0]])
		return xNext.tolist()
