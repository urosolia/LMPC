import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy
import datetime

def main():	
	# parameter initialization
	N = 5
	outfile = TemporaryFile()
	ftocp = FTOCP(N=N) # ftocp used by LMPC
	itMax = 10 # max number of itertions
	itCounter = 1 # iteration counter 
	x0 = [0, 0, 0] # initial condition


	# Compute feasible trajectory
	xclFeasible, uclFeasible = feasTraj(ftocp, 101, ftocp.radius)
	print np.round(xclFeasible, decimals=2)
	np.savetxt('storedData/closedLoopFeasible.txt',xclFeasible.T, fmt='%f' )

	# Initialize LMPC object
	# lmpc = LMPC(ftocp, l=10, P = 200, verbose = False)
	# lmpc = LMPC(ftocp, l=4, P = 50, verbose = False)
	# lmpc = LMPC(ftocp, l=3, P = 25, verbose = False)
	lmpc = LMPC(ftocp, l=1, P = 12, verbose = False)

 	# Add feasible trajectory to the safe set
	lmpc.addTrajectory(xclFeasible, uclFeasible)
	lmpc.ftocp.set_xf(xclFeasible[:,-1])
	# Iteration loop
	meanTimeCostLMPC = []
	while itCounter <= itMax:
		time = 0
		itFlag = 0
		xcl = [x0]
		ucl = []
		timeLMPC = []
		# Time loop
		while (itFlag == 0):
			xt = xcl[time] # read measurement

			startTimer = datetime.datetime.now()
			lmpc.solve(xt, verbose = 1) # solve LMPC
			deltaTimer = datetime.datetime.now() - startTimer
			timeLMPC.append(deltaTimer.total_seconds())

			# Apply input and store closed-loop data
			ut = lmpc.ut
			ucl.append(copy.copy(ut))
			xcl.append(copy.copy(ftocp.f(xcl[time], ut)))

			# Print results
			if lmpc.verbose == True:
				print "State trajectory at time ", time
				print np.round(np.array(xcl).T, decimals=2)
				print np.round(np.array(ucl).T, decimals=2)
				print "==============================================="

			# Check if goal state has been reached
			if np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) <= 1e-4:
				if lmpc.verbose == True:
					print "Distance from terminal point:", xcl[-1]-xclFeasible[:,-1]
				break

			# increment time counter
			time += 1

		# iteration completed. Add trajectory to the safe set
		xcl = np.array(xcl).T
		ucl = np.array(ucl).T
		lmpc.addTrajectory(xcl, ucl) 

		# Store time and cost
		meanTimeCostLMPC.append(np.array([np.sum(timeLMPC)/lmpc.cost, lmpc.cost]))

		# Print and store results
		print "++++++===============================================++++++"
		print "Completed Iteration: ", itCounter
		print "++++++===============================================++++++"
		np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(xcl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/inputIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(ucl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/meanTimeLMPC_P_'+str(lmpc.P)+'.txt', np.array(meanTimeCostLMPC), fmt='%f' )
		np.save(outfile, xcl)

		itCounter += 1 # increment iteration counter
	


def feasTraj(ftocp, timeSteps, radius):
	# Compute first feasible trajectory

	# Intial condition
	xcl = [np.array([0.0,0.0,0.0])]
	ucl =[]
	u = [0,0]

	# Simple brute force hard coded if logic
	for i in range(0, timeSteps):
		if i ==0:
			u[1] =  0.25
		elif i== 1:
			u[1] =   0.25
		elif i== 2:
			u[1] =   0.25
		elif i==(timeSteps-4):
			u[1] =  -0.25
		elif i==(timeSteps-3):
			u[1] =  -0.25
		elif i==(timeSteps-2):
			u[1] =  -0.25
		else:
			u[1] = 0   
		
		u[0] =  xcl[-1][0] / radius;

		xcl.append(ftocp.f(xcl[-1], u))
		ucl.append(np.array(u))


	return np.array(xcl).T[:,:-1], np.array(ucl).T[:,:-1]



if __name__== "__main__":
  main()