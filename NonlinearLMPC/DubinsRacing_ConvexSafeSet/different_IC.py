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

def main():
	# Check if a storedData folder exist.	
	if not os.path.exists('storedData'):
		os.makedirs('storedData')

	# parameter initialization
	N = 4
	roadHalfWidth = 2.0
	outfile = TemporaryFile()
	ftocp = FTOCP(N=N,roadHalfWidth=roadHalfWidth) # ftocp used by LMPC
	itMax = 10 # max number of itertions
	itCounter = 1 # iteration counter 
	x0 = [[ 0.15,-0.1, 0],
		  [ 0.1,  0.3, 0],
		  [ 0.5,  0.5, 0],
		  [0.25, 0.25, 0],
		  [0.25,-0.25, 0],
		  [ 0.0, 0.0, 0],
		  [ 0.5, 0, 0],
		  [ 0, 0.25, 0],
		  [ 0.25, 0.0, 0],
		  [ 0.15, 0.2, 0],] # initial condition


	# Initialize LMPC object
	lmpc = LMPC(ftocp, l=6, P = 20, verbose = False)


	xit = []
	P = 200;
	for i in range(1,8):
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
		ucl = np.loadtxt('storedData/inputIteration'+str(i)+'_P_'+str(P)+'.txt')
		xcl = np.array(xcl).T
		ucl = np.array(ucl).T
		# Add feasible trajectory to the safe set
		lmpc.addTrajectory(xcl, ucl)

	xclFeasible = xcl
	
	# Pick terminal state or terminal set
	terminalSet = True
	if terminalSet == False:
		lmpc.ftocp.set_xf(np.array([xclFeasible[:,-1]]).T)
	else:	
		sFinishLine = xclFeasible[0,-1]
		delta_s = 0.5 #2.0
		Xf = np.array([[sFinishLine, -roadHalfWidth, 0.0],[sFinishLine+delta_s, roadHalfWidth, 0.0]]).T
		print "Box Xf"	
		print Xf
		lmpc.ftocp.set_xf(Xf)

		Xf_vertices = np.concatenate((Xf,Xf), axis = 1 )
		Xf_vertices[1,2] = roadHalfWidth
		Xf_vertices[1,3] = -roadHalfWidth
		lmpc.Xf_vertices = Xf_vertices
		print "Verices Xf"	
		print Xf_vertices


	meanTimeCostLMPC = []
	for idxIC in range(0, len(x0)):
		time = 0
		itFlag = 0
		xcl = [x0]
		ucl = []
		timeLMPC = []

		xcl = [x0[idxIC]]
		print('x0: ', xcl)
		while (itFlag == 0):
			xt = xcl[time] # read measurement
			print("Time: ", time, " xt: ", xt)

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
			if lmpc.ftocp.checkTaskCompletion(xcl[-1]):
			# if np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) <= 1e-4:
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
		np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'_IC.txt', np.round(np.array(xcl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/inputIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'_IC.txt', np.round(np.array(ucl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/meanTimeLMPC_P_'+str(lmpc.P)+'_IC.txt', np.array(meanTimeCostLMPC), fmt='%f' )
		np.save(outfile, xcl)

		itCounter += 1 # increment iteration counter
	


def feasTraj(ftocp, timeSteps, x0):
	# Compute first feasible trajectory

	# Intial condition
	xcl = [np.array(x0)]
	ucl =[]
	u = [0]

	# Simple brute force hard coded if logic
	for i in range(0, timeSteps):
		if i ==0:
			u[0] =  0.5
		elif xcl[-1][0] + ftocp.dt*xcl[-1][1] == 0:
			u[0] = -xcl[-1][1]/(ftocp.dt*(1-(xcl[-1][1]/ ftocp.maxV)**2) )
		else:
			u[0] = 0   
		
		xcl.append(ftocp.f(xcl[-1], u))
		ucl.append(np.array(u))

	return np.array(xcl).T[:,:-1], np.array(ucl).T[:,:-1]



if __name__== "__main__":
  main()