import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy

def main():	
	# parameter initialization
	N = 4
	outfile = TemporaryFile()
	tollTrajectory = 5
	ftocp = FTOCP(N=N) # ftocp used by LMPC
	itMax = 10 # max number of itertions
	itCounter = 1 # iteration counter 
	x0 = [0, 0, 0] # initial condition


	# Compute feasible trajectory
	xclFeasible, uclFeasible = feasTraj(ftocp, 40, np.pi/6*2)
	xclFeasible = np.round(xclFeasible, decimals=tollTrajectory)
	uclFeasible = np.round(uclFeasible, decimals=tollTrajectory)
	print np.round(xclFeasible, decimals=2)
	np.savetxt('storedData/closedLoopFeasible.txt',xclFeasible, fmt='%f' )

	# Initialize LMPC object
	lmpc = LMPC(ftocp, l=4, M=100) # Basically use all SS
	lmpc = LMPC(ftocp, l=2, M=10) # Converges also with small SS
	# lmpc = LMPC(ftocp, l=2, M=8)  # Converges but cost not monotonically decreasing
 	# lmpc = LMPC(ftocp, l=2, M=6)  # Does not converges

 	# Add feasible trajectory to the safe set
	lmpc.addTrajectory(xclFeasible, uclFeasible)

	# Iteration loop
	while itCounter <= itMax:
		time = 0
		itFlag = 0
		xcl = [x0]
		ucl = []
		# Time loop
		while (itFlag == 0):
			xt = xcl[time] # read measurement

			lmpc.solve(xt, verbose = 1) # solve LMPC

			# Apply input and store closed-loop data
			ut = lmpc.ut
			ucl.append(copy.copy(ut))
			xcl.append(copy.copy(ftocp.f(xcl[time], ut)))

			# Print results
			print "State trajectory at time ", time
			print np.round(np.array(xcl).T, decimals=2)
			print np.round(np.array(ucl).T, decimals=2)
			print "==============================================="

			# Check if goal state has been reached
			if np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) <= 1e-4:
				print xcl[-1]-xclFeasible[:,-1]
				break
			elif lmpc.ftocp.N == 0:
				# pdb.set_trace()
				break

			# increment time counter
			time += 1

		# iteration completed. Add trajectory to the safe set
		ftocp.buildNonlinearProgram(N=N) # reset controller horizon
		xcl = np.round(np.array(xcl).T, decimals=tollTrajectory)
		ucl = np.round(np.array(ucl).T, decimals=tollTrajectory)
		lmpc.addTrajectory(xcl, ucl) 

		# Print and store results
		print "++++++===============================================++++++"
		print "Completed Iteration: ", itCounter
		print "++++++===============================================++++++"
		np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'.txt',np.round(np.array(xcl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/inputIteration'+str(itCounter)+'.txt',np.round(np.array(ucl), decimals=5).T, fmt='%f' )
		np.save(outfile, xcl)

		itCounter += 1 # increment iteration counter
	


def feasTraj(ftocp, timeSteps, angle):
	# Compute first feasible trajectory

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