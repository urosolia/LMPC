import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy
import os
import datetime

def main():
	# Check if a storedData folder exist.	
	if not os.path.exists('storedData'):
	    os.makedirs('storedData')

	# parameter initialization
	N = 6 # controller horizon
	outfile = TemporaryFile() # Open temp file to store trajectory
	ftocp = FTOCP(N=N) # ftocp used by LMPC
	itMax = 6 # max number of itertions
	itCounter = 1 # iteration counter 
	x0 = [0, 0, 0] # initial condition


	# Compute feasible trajectory
	xclFeasible, uclFeasible = feasTraj(ftocp, 40, np.pi/6*2)
	print np.round(xclFeasible, decimals=2)
	np.savetxt('storedData/closedLoopFeasible.txt',xclFeasible, fmt='%f' )

	# Initialize LMPC object
	safeSetOption = 'timeVarying' # Allowed options are 'timeVarying', 'spaceVarying' and 'all'

	# lmpc = LMPC(ftocp, l='all', P='all', safeSetOption='all') # Basically this is the LMPC from the TAC paper as uses all iteration and data points
	# lmpc = LMPC(ftocp, l=3, P=40, safeSetOption=safeSetOption) 
	# lmpc = LMPC(ftocp, l=3, P=16, safeSetOption=safeSetOption) 
	# lmpc = LMPC(ftocp, l=2, P=10, safeSetOption=safeSetOption)
	lmpc = LMPC(ftocp, l=1,  P=8, safeSetOption=safeSetOption)

 	# Add feasible trajectory to the safe set
	lmpc.addTrajectory(xclFeasible, uclFeasible)

	# Iteration loop
	meanTimeCostLMPC = []
	while itCounter <= itMax:
		timeStep = 0 # time counter
		xcl = [x0] # closed-loop iteration j
		ucl = [] # input iteration j
		timeLMPC = [] # Computational cost at each time t
		# Time loop
		while (np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) >= 1e-4):
			xt = xcl[timeStep] # read measurement

			startTimer = datetime.datetime.now()
			lmpc.solve(xt, verbose = 1) # solve LMPC
			deltaTimer = datetime.datetime.now() - startTimer
			timeLMPC.append(deltaTimer.total_seconds()) # store LMPC computational time

			# Apply input and store closed-loop data
			ut = lmpc.ut
			ucl.append(copy.copy(ut))
			xcl.append(copy.copy(ftocp.f(xcl[timeStep], ut)))

			# Print results
			print "State trajectory at time ", timeStep
			print np.round(np.array(xcl).T, decimals=2)
			print np.round(np.array(ucl).T, decimals=2)
			print "==============================================="

			# increment time counter
			timeStep += 1

		# iteration completed. Add trajectory to the safe set
		xcl = np.array(xcl).T
		ucl = np.array(ucl).T 
		lmpc.addTrajectory(xcl, ucl) 

		# store mean LMPC time and iteration cost
		meanTimeCostLMPC.append(np.array([np.sum(timeLMPC)/lmpc.cost, lmpc.cost])) 
		
		# Print and store results
		print "++++++===============================================++++++"
		print "Completed Iteration: ", itCounter
		print "++++++===============================================++++++"

		# Save closed-loop, input and stats to .txt files
		np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(xcl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/inputIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(ucl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/meanTimeLMPC_P_'+str(lmpc.P)+'.txt', np.array(meanTimeCostLMPC), fmt='%f' )
		np.save(outfile, xcl)

		itCounter += 1 # increment iteration counter
	


def feasTraj(ftocp, timeSteps, angle):
	# Compute first feasible trajectory: hand crafted algorithm

	# Intial condition
	xcl = [np.array([0.0,0.0,0.0])]
	ucl =[]
	u = [0,0]

	# Simple brute force hard coded if logic
	for i in range(0, timeSteps):
		if i ==0:
			u[1] =  1
		elif i== 1:
			u[1] =   1
		elif i== 2:
			u[1] =   1
		elif i==(timeSteps-4):
			u[1] =  -1
		elif i==(timeSteps-3):
			u[1] =  -1
		elif i==(timeSteps-2):
			u[1] =  -1
		else:
			u[1] = 0   
	
		if i<(timeSteps/2) and i > 0:
			u[0] =  angle;
		elif i >= (timeSteps/2): 
			u[0] = -angle;

		xcl.append(ftocp.f(xcl[-1], u))
		ucl.append(np.array(u))


	return np.array(xcl).T[:,:-1], np.array(ucl).T[:,:-1]

if __name__== "__main__":
  main()