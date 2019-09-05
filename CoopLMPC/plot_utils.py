import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import copy, pickle, pdb, time
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Trajectory animation
def plot_agent_trajs(x, deltas=None, r=0, trail=False):
    n_a = len(x)
    traj_lens = [x[i].shape[1] for i in range(n_a)]
    end_flags = [False for i in range(n_a)]

    c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([-1.5, 2.5])
    ax.set_ylim([-1.5, 1.5])

    t = 0
    while not np.all(end_flags):
        if not trail:
            ax.clear()
            ax.set_xlim([-1.5, 2.5])
            ax.set_ylim([-1.5, 1.5])
        for i in range(n_a):
            plot_t = min(t, traj_lens[i]-1)
            ax.plot(x[i][0,plot_t], x[i][1,plot_t], '.', c=c[i])
            if r > 0:
                ax.plot(x[i][0,plot_t]+r*np.cos(np.linspace(0,2*np.pi,100)),
                    x[i][1,plot_t]+r*np.sin(np.linspace(0,2*np.pi,100)), c=c[i])
                # ax.plot(x[i][0,plot_t]+l*np.array([-1, -1, 1, 1, -1]), x[i][1,plot_t]+l*np.array([-1, 1, 1, -1, -1]), c=c[i])
            if deltas is not None:
                ax.plot(x[i][0,plot_t]+deltas[t][i]*np.cos(np.linspace(0,2*np.pi,100)),
                    x[i][1,plot_t]+deltas[t][i]*np.sin(np.linspace(0,2*np.pi,100)), '--', c=c[i])
            if not end_flags[i] and t >= traj_lens[i]-1:
                end_flags[i] = True
        t += 1
        fig.canvas.draw()
        time.sleep(0.02)
    plt.ioff()

def plot_ts(x, title=None, x_label=None, y_labels=None):
    plt.figure()
    for i in range(x.shape[0]):
        plt.subplot(x.shape[0], 1, i+1)
        plt.plot(range(x.shape[1]), x[i,:])
        if i == 0 and title is not None:
            plt.title(title)
        if i == x.shape[0]-1 and x_label is not None:
            plt.xlabel(x_label)
        if y_labels is not None:
            plt.ylabel(y_labels[i])
