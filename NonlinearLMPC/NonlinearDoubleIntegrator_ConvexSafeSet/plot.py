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
# 0 = blue, 1 = orange, 2 = green, 3 = red, 4 = purple
colorMap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# colorMap = ["#3182bd", "#fd8d3c", "#31a354", "#e6550d", "#756bb1"]

xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
xFeasible = xFeasible.T
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[1,:], '-d', color=colorMap[2], label='Feasible trajectory')
iterationTime.append(xFeasible.shape[1]-1) # Store time to reach xf


xit = []
for i in range(1,it):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'_P_'+str(P)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[1,:], 's', color=colorMap[3]) 
	iterationTime.append(xcl.shape[1]-1) # Store time to reach xf


plt.plot(0, 0, 's', color=colorMap[3], label='Stored data')
xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P)+'.txt')
xcl = xcl.T
iterationTime.append(xcl.shape[1]-1) # Store time to reach xf
xit.append(copy.copy(xcl))
plt.plot(xcl[0,:], xcl[1,:], 's', color=colorMap[3]) 
plt.plot(xcl[0,:], xcl[1,:], '-o', color=colorMap[0], label='LMPC closed-loop at '+str(it)+'th iteration')


plt.xlabel('$z$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.legend()
print iterationTime

# =========================================================
# Plot input commands
# =========================================================

plt.figure()
ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P)+'.txt')
plt.plot(ucl[:], '-o', color=colorMap[0], label="LMPC closed-loop for P = "+str(P))
plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)

plt.legend()

# =========================================================
# Plot computational time
# =========================================================

plt.figure()
mat = np.loadtxt('storedData/meanTimeLMPC'+'_P_'+str(P)+'.txt')
compTime = mat[:,0].tolist()
plt.plot(range(1,len(compTime)+1), compTime, '-o', color=colorMap[0], label='${P =}$'+str(P))		

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
P = [12,20,50,200]
if input == 'y':
	i = it
	plt.figure()

	for i in range(0,len(P)):
		ucl = np.loadtxt('storedData/inputIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		plt.plot(ucl[:], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))
		plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)

	plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
	plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
	plt.xlabel('$\mathrm{Time~Step}$', fontsize=20)
	plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
	plt.legend()

	# =========================================================
	# Closed-loop comparison
	# =========================================================
	plt.figure()
	xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
	xFeasible = xFeasible.T
	plt.plot(xFeasible[0,:], xFeasible[1,:], '-dg', label='Feasible trajectory')

	for i in range(0, len(P)):
		xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'_P_'+str(P[i])+'.txt')
		xcl = xcl.T
		plt.plot(xcl[0,:], xcl[1,:], '-o', label="LMPC closed-loop for P = "+str(P[i])+", i="+str(l[i]))

	plt.ylim([0,4])
	plt.xlabel('$x$', fontsize=20)
	plt.ylabel('$v$', fontsize=20)

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