import numpy as np 
import matplotlib.pyplot as plt
import copy
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


it = 10
iterationTime = []
# =========================================================
# Plot closed-loop
# =========================================================

xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[1,:], '-dg', label='Feasible trajectory')
iterationTime.append(xFeasible.shape[1]-1) # Store time to reach xf

print xFeasible

xit = []
for i in range(1,it):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[1,:], 'sr') 
	iterationTime.append(xcl.shape[1]-1) # Store time to reach xf


plt.plot(0, 0, 'sr', label='Stored data')
xcl = np.loadtxt('storedData/closedLoopIteration'+str(it)+'.txt')
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
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.legend()

# =========================================================
# Plot velocity
# =========================================================
xFeasible = np.loadtxt('storedData/closedLoopFeasible.txt')
plt.figure()
plt.plot(xFeasible[0,:], xFeasible[2,:], '-dg', label='Feasible trajectory')

xit = []
for i in range(1,it+1):
	xcl = np.loadtxt('storedData/closedLoopIteration'+str(i)+'.txt')
	xcl = xcl.T
	xit.append(copy.copy(xcl))
	plt.plot(xcl[0,:], xcl[2,:], 'sr')
plt.plot(0, 0, 'sr', label='Stored data')

plt.plot(xcl[0,:], xcl[2,:], '-ob', label='LMPC closed-loop at '+str(it)+'th iteration')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$\mathrm{velocity}$', fontsize=20)
plt.legend()

# =========================================================
# Plot inputs
# =========================================================
i = it
ucl = np.loadtxt('storedData/inputIteration'+str(i)+'.txt')
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(ucl[:,0], '-ob')
plt.ylabel('$\mathrm{Steering}$', fontsize=20)

plt.subplot(2, 1, 2)
plt.plot(ucl[:,1], '-ob')
plt.plot([0,ucl.shape[0]-1],[1,1], '--k', label='Saturation limit')
plt.plot([0,ucl.shape[0]-1],[-1,-1], '--k')
plt.xlabel('$\mathrm{Time}$', fontsize=20)
plt.ylabel('$\mathrm{Acceleration}$', fontsize=20)
plt.legend()

plt.show()