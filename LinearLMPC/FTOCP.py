import numpy as np
import pdb 
import scipy
from cvxpy import *
import itertools

class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""
	def __init__(self, N, A, B, Q, R):
		# Define variables
		self.N = N # Horizon Length

		# System Dynamics (x_{k+1} = A x_k + Bu_k)
		self.A = A 
		self.B = B 
		self.n = A.shape[1]
		self.d = B.shape[1]

		# Cost (h(x,u) = x^TQx +u^TRu)
		self.Q = Q
		self.R = R

		# Initialize Predicted Trajectory
		self.xPred = []
		self.uPred = []

	def solve(self, x0, verbose = 0, SS =[], Qfun=[], CVX =[]):
		"""This methos solve a FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		""" 

		# Initialize Variables
		x = Variable((self.n, self.N+1))
		u = Variable((self.d, self.N))

		# If SS is given construct a matrix collacting all states and a vector collection all costs
		if SS != []:
			SS_vector = np.squeeze(list(itertools.chain.from_iterable(SS))).T # From a 3D list to a 2D array
			Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(Qfun))), 0) # From a 2D list to a 1D array
			if CVX == True:
				lambVar = Variable((SS_vector.shape[1], 1), boolean=False) # Initialize vector of variables
			else:
				lambVar = Variable((SS_vector.shape[1], 1), boolean=True) # Initialize vector of variables

		# State Constraints
		constr = [x[:,0] == x0[:]]
		for i in range(0, self.N):
			constr += [x[:,i+1] == self.A*x[:,i] + self.B*u[:,i],
						u[:,i] >= -1.0,
						u[:,i] <=  1.0,
						x[:,i] >= -10.0,
						x[:,i] <=  10.0,]

		# Terminal Constraint if SS not empty
		if SS != []:
			constr += [SS_vector * lambVar[:,0] == x[:,self.N], # Terminal state \in ConvHull(SS)
						np.ones((1, SS_vector.shape[1])) * lambVar[:,0] == 1, # Multiplies \lambda sum to 1
						lambVar >= 0] # Multiplier are positive definite

		# Cost Function
		cost = 0
		for i in range(0, self.N):
			cost += norm(self.Q**0.5*x[:,i])**2 + norm(self.R**0.5*u[:,i])**2 # Running cost h(x,u) = x^TQx + u^TRu

		# Terminal cost if SS not empty
		if SS != []:
			cost += Qfun_vector[0,:] * lambVar[:,0]  # It terminal cost is given by interpolation using \lambda
		else:
			cost += norm(self.Q**0.5*x[:,self.N])**2 # If SS is not given terminal cost is quadratic


		# Solve the Finite Time Optimal Control Problem
		problem = Problem(Minimize(cost), constr)
		problem.solve(verbose=verbose==1)

		# Store the open-loop predicted trajectory
		self.xPred = x.value
		self.uPred = u.value	


	def model(self, x, u):
		# Compute state evolution
		return (np.dot(self.A,x) + np.squeeze(np.dot(self.B,u))).tolist()





	

