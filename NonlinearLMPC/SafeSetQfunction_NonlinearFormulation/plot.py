import numpy as np 
import matplotlib.pyplot as plt
import copy
from matplotlib import rc
import pdb
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


it = 10
iterationTime = []
circleRadius = 10.0
P = 12
print "Current value of P = ",P
# a = raw_input("Pick Value of a = ")

# =========================================================
# Plot closed-loop
# =========================================================

xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
xFeasible = xFeasible.T
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[1,:], '-dg', label='Feasible trajectory')
iterationTime.append(xFeasible.shape[1]-1) # Store time to reach xf


xit = []
for i in range(1,it):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[1,:], 'sr') 
	iterationTime.append(xcl.shape[1]-1) # Store time to reach xf


plt.plot(0, 0, 'sr', label='Stored data')
xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P)+'.txt')
xcl = xcl.T
iterationTime.append(xcl.shape[1]-1) # Store time to reach xf
xit.append(copy.copy(xcl))
plt.plot(xcl[0,:], xcl[1,:], 'sr') # Store time to reach xf
plt.plot(xcl[0,:], xcl[1,:], '-ob', label='LMPC closed-loop at '+str(it)+'th iteration')

print iterationTime

plt.xlabel('$z$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.legend()
# =========================================================
# Plot in x-y
# ===================================================1======
plt.figure()
pointCircleIn =[]
pointCircleOut =[]
feasibleTraj = []

for k in range(0, np.shape(xFeasible)[1]):
	angle  = xFeasible[0,k]/circleRadius
	radius = circleRadius  - xFeasible[1,k]
	radiusIn  = circleRadius - 0.3
	radiusOut = circleRadius + 0.3
	pointCircleIn.append([radiusIn*np.sin(angle), circleRadius-radiusIn*np.cos(angle)])
	pointCircleOut.append([radiusOut*np.sin(angle), circleRadius-radiusOut*np.cos(angle)])

	feasibleTraj.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

plt.plot(np.array(feasibleTraj)[:,0], np.array(feasibleTraj)[:,1], '-dg', label='Feasible trajectory')
	
pointCircleInArray = np.array(pointCircleIn)
plt.plot(pointCircleInArray[:,0], pointCircleInArray[:,1], '-k', label='Road Boundaries')
pointCircleOutArray = np.array(pointCircleOut)
plt.plot(pointCircleOutArray[:,0], pointCircleOutArray[:,1], '-k')


xit = []
for i in range(it-1,it):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	xclTot = []
	for k in range(0, np.shape(xcl)[1]):
		angle  = xcl[0,k]/circleRadius
		radius = circleRadius- xcl[1,k]
		xclTot.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])
	plt.plot(np.array(xclTot)[:,0], np.array(xclTot)[:,1], 'sr')

plt.plot(np.array(xclTot)[:,0], np.array(xclTot)[:,1], '-ob', label='LMPC closed-loop at '+str(it)+'th iteration')



plt.xlim(-12,13)
plt.ylim(-12,13)
plt.axis('equal')
plt.xlabel('$x_{(1)}$', fontsize=20)
plt.ylabel('$x_{(2)}$', fontsize=20)
plt.legend()


# =========================================================
# Plot velocity
# =========================================================
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[2,:], '-dg', label='Feasible trajectory')

xit = []
for i in range(1,it+1):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[2,:], 'sr')
plt.plot(0, 0, 'sr', label='Stored data')

plt.plot(xcl[0,:], xcl[2,:], '-ob', label='LMPC closed-loop at '+str(it)+'th iteration')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$\mathrm{velocity}$', fontsize=20)
plt.legend()
plt.xlim(0,19)
plt.ylim(0,6)


plt.figure()

plt.subplot(2, 1, 1)	
ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P)+'.txt')
plt.plot(ucl[:,0], '-o', label="LMPC closed-loop for P = "+str(P))
plt.ylabel('$\mathrm{Steering}$', fontsize=20)

plt.legend()

plt.subplot(2, 1, 2)
ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P)+'.txt')
plt.plot(ucl[:,1], '-o')#, label="LMPC closed-loop for P = "+str(i)+", i="+str(l[counter]))

plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
plt.xlabel('$\mathrm{Time~Step}$', fontsize=20)
plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
plt.legend()	

plt.figure()
mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P)+'.txt')
compTime = mat[:,0].tolist()
plt.plot(range(1,len(compTime)+1), compTime, '-o', label='${P =}$'+str(P))		

plt.xlabel('$\mathrm{Iteration~}j$', fontsize=20)
plt.ylabel('$\mathrm{Mean~Computational~Time~[s]}$', fontsize=20)
plt.legend()

# =========================================================
# Run Comparison
# =========================================================
input = raw_input("Do you want to run comparison for different values of P and l? [y/n] ")

# =========================================================
# Plot inputs
# =========================================================
l = [1, 2,3, 10]
P = [12,25,50,200]
if input == 'y':
	i = it
	plt.figure()

	plt.subplot(2, 1, 1)	
	for i in range(0,len(P)):
		ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		plt.plot(ucl[:,0], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))
		plt.ylabel('$\mathrm{Steering}$', fontsize=20)

	plt.legend()

	plt.subplot(2, 1, 2)
	for i in range(0,len(P)):
		ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		plt.plot(ucl[:,1], '-o')#, label="LMPC closed-loop for P = "+str(i)+", i="+str(l[counter]))

	plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
	plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
	plt.xlabel('$\mathrm{Time~Step}$', fontsize=20)
	plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
	plt.legend()

	# =========================================================
	# Closed-loop comparison
	# =========================================================
	plt.figure()
	pointCircleIn =[]
	pointCircleOut =[]
	feasibleTraj = []

	for k in range(0, np.shape(xFeasible)[1]):
		angle  = xFeasible[0,k]/circleRadius
		radius = circleRadius  - xFeasible[1,k]
		radiusIn  = circleRadius - 0.3
		radiusOut = circleRadius + 0.3
		pointCircleIn.append([radiusIn*np.sin(angle), circleRadius-radiusIn*np.cos(angle)])
		pointCircleOut.append([radiusOut*np.sin(angle), circleRadius-radiusOut*np.cos(angle)])

		feasibleTraj.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

	plt.plot(np.array(feasibleTraj)[:,0], np.array(feasibleTraj)[:,1], '-dk', label='Feasible trajectory')
		
	pointCircleInArray = np.array(pointCircleIn)
	plt.plot(pointCircleInArray[:,0], pointCircleInArray[:,1], '-k', label='Road Boundaries')
	pointCircleOutArray = np.array(pointCircleOut)
	plt.plot(pointCircleOutArray[:,0], pointCircleOutArray[:,1], '-k')	

	for i in range(0, len(P)):
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		xclTot = []
		xcl = xcl.T
		for k in range(0, xcl.shape[1]):
			angle  = xcl[0,k]/circleRadius
			radius = circleRadius- xcl[1,k]
			xclTot.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

		xclTotArray = np.array(xclTot)
		plt.plot(xclTotArray[:,0], xclTotArray[:,1], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))

	plt.xlabel('$x$', fontsize=20)
	plt.ylabel('$y$', fontsize=20)

	plt.legend()
	
	plt.figure()
	pointCircleIn =[]
	pointCircleOut =[]
	feasibleTraj = []

	for k in range(0, np.shape(xFeasible)[1]):
		angle  = xFeasible[0,k]/circleRadius
		radius = circleRadius  - xFeasible[1,k]
		radiusIn  = circleRadius - 0.3
		radiusOut = circleRadius + 0.3
		pointCircleIn.append([radiusIn*np.sin(angle), circleRadius-radiusIn*np.cos(angle)])
		pointCircleOut.append([radiusOut*np.sin(angle), circleRadius-radiusOut*np.cos(angle)])

		feasibleTraj.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

	plt.plot(np.array(feasibleTraj)[:,0], np.array(feasibleTraj)[:,1], '-dk', label='Feasible trajectory')
		
	pointCircleInArray = np.array(pointCircleIn)
	plt.plot(pointCircleInArray[:,0], pointCircleInArray[:,1], '-k', label='Road Boundaries')
	pointCircleOutArray = np.array(pointCircleOut)
	plt.plot(pointCircleOutArray[:,0], pointCircleOutArray[:,1], '-k')	

	for i in range(0, len(P)):
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		xclTot = []
		xcl = xcl.T
		for k in range(0, xcl.shape[1]):
			angle  = xcl[0,k]/circleRadius
			radius = circleRadius- xcl[1,k]
			xclTot.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

		xclTotArray = np.array(xclTot)
		plt.plot(xclTotArray[:,0], xclTotArray[:,1], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))

	plt.xlabel('$x$', fontsize=20)
	plt.ylabel('$y$', fontsize=20)
	plt.axis('equal')

	plt.legend()

	# =========================================================
	# Time and computational cost
	# =========================================================
	plt.figure()
	for i in range(0,len(P)):
		mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P[i])+'.txt')
		cost = mat[:,1].tolist()
		cost.insert(0, 100)
		plt.plot(range(0,len(cost)), cost, '-o', label='${P =}$'+str(P[i])+', ${i =}$'+str(l[i]))
	
	plt.xlabel('$\mathrm{Iteration~}j$', fontsize=20)
	plt.ylabel('$\mathrm{Time~Steps~}T^j$', fontsize=20)
	plt.legend()

	plt.figure()
	for i in range(0, len(P)):
		mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P[i])+'.txt')
		compTime = mat[:,0].tolist()
		plt.plot(range(1,len(compTime)+1), compTime, '-o', label='${P =}$'+str(P[i])+', ${i =}$'+str(l[i]))		

	plt.xlabel('$\mathrm{Iteration~}j$', fontsize=20)
	plt.ylabel('$\mathrm{Mean~Computational~Time~[s]}$', fontsize=20)
	plt.legend()

	plt.show()

plt.show()