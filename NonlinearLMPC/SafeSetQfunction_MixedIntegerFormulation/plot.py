import numpy as np 
import matplotlib.pyplot as plt
import copy
from matplotlib import rc
import pdb
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


it = 6
iterationTime = []
P = 8
print "Number of points used: ", P
# =========================================================
# Plot closed-loop
# =========================================================
xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[1,:], '-dk', label='Feasible trajectory')
iterationTime.append(xFeasible.shape[1]-1) # Store time to reach xf

print xFeasible

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

x_obs = []
y_obs = []
for i in np.linspace(0,2*np.pi,1000):
	x_obs.append(27 + 8*np.cos(i))
	y_obs.append(-1 + 6*np.sin(i))

plt.plot(x_obs, y_obs, '-k', label='Obstacle')
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.legend()

# =========================================================
# Plot velocity and acceleration
# =========================================================
i = it
ucl = np.loadtxt('storedData/inputIteration'+str(i)+'_P_'+str(P)+'.txt')
xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(xcl[:,2], '-ob')
plt.ylabel('$\mathrm{Velocity}$', fontsize=20)

plt.subplot(2, 1, 2)
plt.plot(ucl[:,1], '-ob')
plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
plt.xlabel('$\mathrm{Time~Step}$', fontsize=20)
plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
plt.legend()

# =========================================================
# Plot velocity
# =========================================================
xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[2,:], '-dg', label='Feasible trajectory')

xit = []
for i in range(1,it+1):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[2,:], 'sr')
plt.plot(0, 0, 'sr', label='Stored data')

plt.plot(xcl[0,:], xcl[2,:], '-ob', label='LMPC closed-loop')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$\mathrm{velocity}$', fontsize=20)
plt.legend()

# =========================================================
# Plot inputs
# =========================================================
i = it
ucl = np.loadtxt('storedData/inputIteration'+str(i)+'_P_'+str(P)+'.txt')
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(ucl[:,0], '-ob')
plt.ylabel('$\mathrm{Steering}$', fontsize=20)

plt.subplot(2, 1, 2)
plt.plot(ucl[:,1], '-ob')
plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
plt.xlabel('$\mathrm{Time~Step}$', fontsize=20)
plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
plt.legend()

# =========================================================
# Run Comparison
# =========================================================
input = raw_input("Do you want to run comparison for different values of P and l? [y/n] ")

# =========================================================
# Plot inputs
# =========================================================
l = [1, 2,3, 10]
P = [8,10,16,'all']
labelLMPC = 'LMPC from [21]'
pltMarker = ['-o', '--s','-d','--v']
if input == 'y':
	i = it
	plt.figure()

	plt.subplot(2, 1, 1)	
	for i in range(0,len(P)):
		ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		if P[i] == 'all':
			plt.plot(ucl[:,0], '-o', label=labelLMPC)
		else:
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
	# Closed-loop comparison (X-Y)
	# =========================================================
	xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
	plt.figure()
	plt.plot(xFeasible[0,:], xFeasible[1,:], '-dk', label='Feasible trajectory')
	iterationTime.append(xFeasible.shape[1]-1) # Store time to reach xf

	for i in range(0, len(P)):
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		xcl = xcl.T
		if P[i] == "all":
			plt.plot(xcl[0,:], xcl[1,:], '-o', label=labelLMPC)
		else:	
			plt.plot(xcl[0,:], xcl[1,:], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))

	x_obs = []
	y_obs = []
	for i in np.linspace(0,2*np.pi,1000):
		x_obs.append(27 + 8*np.cos(i))
		y_obs.append(-1 + 6*np.sin(i))

	plt.plot(x_obs, y_obs, '-k', label='Obstacle')
	plt.xlabel('$x$', fontsize=20)
	plt.ylabel('$y$', fontsize=20)

	plt.legend()

	# =========================================================
	# Time and computational cost
	# =========================================================
	plt.figure()
	for i in range(0,len(P)):
		mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P[i])+'.txt')
		cost = mat[:,1].tolist()
		cost.insert(0, 39)
		if P[i] == "all":
			plt.plot(range(0,len(cost)), cost, '-o', label=labelLMPC)
		else:
			plt.plot(range(0,len(cost)), cost, '-o', label='${P =}$'+str(P[i])+', ${i =}$'+str(l[i]))
	
	plt.xlabel('$\mathrm{Iteration~}j$', fontsize=20)
	plt.ylabel('$\mathrm{Time~Steps~}T^j$', fontsize=20)
	plt.legend()

	plt.figure()
	for i in range(0, len(P)):
		mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P[i])+'.txt')
		compTime = mat[:,0].tolist()
		if P[i] == "all":
			plt.plot(range(1,len(compTime)+1), compTime, '-o', label=labelLMPC)	
		else:
			plt.plot(range(1,len(compTime)+1), compTime, '-o', label='${P =}$'+str(P[i])+', ${i =}$'+str(l[i]))		

	plt.xlabel('$\mathrm{Iteration~}j$', fontsize=20)
	plt.ylabel('$\mathrm{Mean~Computational~Time~[s]}$', fontsize=20)
	plt.legend()

plt.show()


