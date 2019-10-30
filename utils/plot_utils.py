import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Trajectory animation with circular exploration constraints
def plot_agent_trajs(x, expl_con=None, r_a=None, trail=False, shade=False, plot_lims=None, save_dir=None, save_video=False, it=None):
	dpi = 200

	if expl_con is not None and 'lin' in expl_con:
		H_cl = expl_con['lin'][0]
		g_cl = expl_con['lin'][1]
		boundary_x = np.linspace(plot_lims[0][0], plot_lims[0][1], 50)
		if shade:
			p1 = np.linspace(plot_lims[0][0], plot_lims[0][1], 10)
			p2 = np.linspace(plot_lims[1][0], plot_lims[1][1], 10)
			P1, P2 = np.meshgrid(p1, p2)
	if expl_con is not None and 'ell' in expl_con:
		ell_con = expl_con['ell']

	n_a = len(x)

	if r_a is None:
		r_a = [0 for _ in range(n_a)]

	traj_lens = [x[i].shape[1] for i in range(n_a)]
	end_flags = [False for i in range(n_a)]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	if save_video:
		imgs = []
		fig_a = plt.figure(dpi=dpi)
	else:
		fig_a = []

	plt.ion()
	fig = plt.figure(dpi=dpi)
	ax = fig.gca()
	if plot_lims is not None:
		ax.set_xlim(plot_lims[0])
		ax.set_ylim(plot_lims[1])

	t = 0
	text_vars = []
	while not np.all(end_flags):
		plt.figure(fig.number)
		if len(text_vars) != 0:
			for txt in text_vars:
				txt.remove()
			text_vars = []

		if not trail:
			ax.clear()
			if plot_lims is not None:
				ax.set_xlim(plot_lims[0])
				ax.set_ylim(plot_lims[1])

		for i in range(n_a):
			plot_t = min(t, traj_lens[i]-1)
			ax.plot(x[i][0,plot_t], x[i][1,plot_t], '.', c=c[i])
			text_vars.append(ax.text(x[i][0,plot_t]+r_a[i]+0.05,
				x[i][1,plot_t]+r_a[i]+0.05, str(i+1), fontsize=12,
				bbox=dict(facecolor='white', alpha=1.)))
			if r_a[i] > 0:
				ax.plot(x[i][0,plot_t]+r_a[i]*np.cos(np.linspace(0,2*np.pi,100)),
					x[i][1,plot_t]+r_a[i]*np.sin(np.linspace(0,2*np.pi,100)),
					c=c[i])
				# ax.plot(x[i][0,plot_t]+l*np.array([-1, -1, 1, 1, -1]), x[i][1,plot_t]+l*np.array([-1, 1, 1, -1, -1]), c=c[i])

			if expl_con is not None and 'ell' in expl_con:
				ax.plot(x[i][0,plot_t]+ell_con[i,t]*np.cos(np.linspace(0,2*np.pi,100)),
					x[i][1,plot_t]+ell_con[i,t]*np.sin(np.linspace(0,2*np.pi,100)),
					'--', c=c[i], linewidth=0.7)
			if expl_con is not None and 'lin' in expl_con:
				H = H_cl[i][t]
				g = g_cl[i][t]
				boundary_y = (-H[0,0]*boundary_x - g[0])/(H[0,1]+1e-10)

				if shade:
					for j in range(P1.shape[0]):
						for k in range(P2.shape[1]):
							test_pt = np.array([P1[j,k], P2[j,k]])
							if np.all(H.dot(test_pt) + g <= 0):
								ax.plot(test_pt[0], test_pt[1], '.', c=c[i], markersize=0.5)

				ax.plot(boundary_x, boundary_y, '--', c=c[i])

			if not end_flags[i] and t >= traj_lens[i]-1:
				end_flags[i] = True

		if it is not None:
			ax.set_title('Iteration: %i' % it)
		ax.set_aspect('equal')

		fig.canvas.draw()
		# time.sleep(0.02)
		plt.ioff()

		if save_video:
			width, height = fig.get_size_inches() * fig.get_dpi()
			img_arr = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
			plt.figure(fig_a.number)
			img = plt.imshow(img_arr, animated=True)
			plt.axis('off')
			imgs.append([img])

		t += 1

	if save_video:
		ani = animation.ArtistAnimation(fig_a, imgs, interval=50, blit=True)
		ani.save('/'.join((save_dir, 'it_%i.mp4' % it)), dpi=dpi)

	if save_dir is not None and not save_video:
		fig.savefig('/'.join((save_dir, 'it_%i.png' % it)))

	return fig, fig_a

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

class updateable_plot(object):
	def __init__(self, n_seq, title=None, x_label=None, y_label=None):
		plt.ion()
		self.fig = plt.figure()
		self.ax = plt.gca()
		self.ax.set_xlim([0, 5])
		self.ax.set_ylim([0, 5])

		self.n_seq = n_seq
		self.title = title
		self.x_label = x_label
		self.y_label = y_label

		self.data = [np.empty((2,1)) for _ in range(n_seq)]
		self.c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_seq-1))) for i in range(n_seq)]

	def clear(self):
		self.ax.clear()

	def update(self, d, seq_idx):
		self.data[seq_idx] = np.append(self.data[seq_idx], d, axis=1)
		self.ax.clear()
		for i in range(self.n_seq):
			self.ax.plot(self.data[i][0,:], self.data[i][1,:], '.-', c=c[i])
			if self.title is not None:
				self.set_title(self.title)
			if self.x_label is not None:
				self.set_xlabel(self.x_label)
			if self.y_label is not None:
				self.set_xlabel(self.y_label)

		self.fig.canvas.draw()

class updateable_ts(object):
	def __init__(self, n_seq, title=None, x_label=None, y_label=None):
		plt.ion()
		self.fig = plt.figure()
		self.axs = [self.fig.add_subplot(n_seq, 1, i+1) for i in range(n_seq)]
		for (i, a) in enumerate(self.axs):
			if y_label is not None:
				a.set_ylabel(y_label[i])
			if title is not None and i == 0:
				a.set_title(title)
			if x_label is not None and i == len(self.axs)-1:
				a.set_xlabel(x_label)

		self.fig.canvas.draw()

		self.n_seq = n_seq
		self.title = title
		self.x_label = x_label
		self.y_label = y_label

		# self.data = [np.empty((2,1)) for _ in range(n_seq)]
		# self.c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_seq-1))) for i in range(n_seq)]

	def clear(self):
		for a in self.axs:
			a.clear()

	def update(self, d):
		t = range(d.shape[1])
		for (i, a) in enumerate(self.axs):
			a.plot(t, d[i,:])

		self.fig.canvas.draw()

class lmpc_visualizer(object):
	def __init__(self, pos_dims, n_state_dims, n_act_dims, agent_id, plot_lims=None, plot_dir=None):
		if len(pos_dims) > 2:
			raise(ValueError('Can only plot 2 position dimensions'))
		self.agent_id = agent_id
		self.pos_dims = pos_dims
		self.n_state_dims = n_state_dims
		self.n_act_dims = n_act_dims
		self.plot_dir = plot_dir
		self.plot_lims = plot_lims

		plt.ion()

		# Initialize position plot
		self.pos_fig = plt.figure()
		self.pos_ax = plt.gca()
		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')
		self.pos_ax.set_title('Agent %i' % (agent_id+1))
		self.pos_fig.canvas.set_window_title('agent %i positions' % (agent_id+1))
		self.pos_fig.canvas.draw()

		# Initialize velocity plot
		self.state_fig = plt.figure(figsize=(5,4))
		self.state_axs = [self.state_fig.add_subplot(n_state_dims, 1, i+1) for i in range(n_state_dims)]
		for (i, a) in enumerate(self.state_axs):
			a.set_ylabel('$x_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (agent_id+1))
			if i < len(self.state_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.state_axs)-1:
				a.set_xlabel('$t$')
		self.state_fig.canvas.set_window_title('agent %i states' % (agent_id+1))
		self.state_fig.canvas.draw()

		# Initialize input plot
		self.act_fig = plt.figure(figsize=(5,4))
		self.act_axs = [self.act_fig.add_subplot(n_act_dims, 1, i+1) for i in range(n_act_dims)]
		for (i, a) in enumerate(self.act_axs):
			a.set_ylabel('$u_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (agent_id+1))
			if i < len(self.act_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.act_axs)-1:
				a.set_xlabel('$t$')
		self.act_fig.canvas.set_window_title('agent %i inputs' % (agent_id+1))
		self.act_fig.canvas.draw()

		self.prev_pos_cl = None
		self.prev_state_cl = None
		self.prev_act_cl = None

		self.it = 0

	def set_plot_dir(self, plot_dir):
		self.plot_dir = plot_dir

	def clear_state_plots(self):
		self.pos_ax.clear()
		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')
		self.pos_ax.set_title('Agent %i' % (self.agent_id+1))
		for a in self.state_axs:
			a.clear()
		for (i, a) in enumerate(self.state_axs):
			a.set_ylabel('$x_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (self.agent_id+1))
			if i < len(self.state_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.state_axs)-1:
				a.set_xlabel('$t$')

	def clear_act_plot(self):
		for a in self.act_axs:
			a.clear()
		for (i, a) in enumerate(self.act_axs):
			a.set_ylabel('$u_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (self.agent_id+1))
			if i < len(self.act_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.act_axs)-1:
				a.set_xlabel('$t$')

	def update_prev_trajs(self, state_traj=None, act_traj=None):
		# state_traj is a list of numpy arrays. Each numpy array is the closed-loop trajectory of an agent.
		if state_traj is not None:
			self.prev_pos_cl = [s[self.pos_dims,:] for s in state_traj]
			self.prev_state_cl = state_traj
		if act_traj is not None:
			self.prev_act_cl = act_traj

		self.it += 1

	def plot_state_traj(self, state_cl, state_preds, t, expl_con=None, shade=False):
		self.clear_state_plots()

		if expl_con is not None and 'lin' in expl_con:
			lin_con = expl_con['lin']
		if expl_con is not None and 'ell' in expl_con:
			ell_con = expl_con['ell']

		if shade:
			p1 = np.linspace(self.plot_lims[0][0], self.plot_lims[0][1], 15)
			p2 = np.linspace(self.plot_lims[1][0], self.plot_lims[1][1], 15)
			P1, P2 = np.meshgrid(p1, p2)

		pos_preds = state_preds[self.pos_dims, :]
		pred_len = state_preds.shape[1]

		# Pick out position and velocity dimensions
		pos_cl = state_cl[self.pos_dims, :]
		cl_len = state_cl.shape[1]

		c_pred = [matplotlib.cm.get_cmap('jet')(i*(1./(pred_len-1))) for i in range(pred_len)]

		# Plot entire previous closed loop trajectory for comparison
		if self.prev_pos_cl is not None:

			n_a = len(self.prev_pos_cl)
			c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]
			for (i, s) in enumerate(self.prev_pos_cl):
				plot_t = min(t, s.shape[1]-1)
				self.pos_ax.plot(s[0,:], s[1,:], '.', c=c[i], markersize=1)
				self.pos_ax.plot(s[0,plot_t], s[1,plot_t], 'o', c=c[i])
				# self.pos_ax.text(s[0,0]+0.1, s[1,0]+0.1, 'Agent %i' % (i+1), fontsize=12, bbox=dict(facecolor='white', alpha=1.))

			agent_prev_cl = self.prev_pos_cl[self.agent_id]

			# Plot the ellipsoidal constraint
			if expl_con is not None and 'ell' in expl_con:
				for i in range(t, t+pred_len):
					plot_t = min(i, agent_prev_cl.shape[1]-1)
					self.pos_ax.plot(agent_prev_cl[0,plot_t]+ell_con[plot_t]*np.cos(np.linspace(0,2*np.pi,100)),
						agent_prev_cl[1,plot_t]+ell_con[plot_t]*np.sin(np.linspace(0,2*np.pi,100)), '--', linewidth=0.7, c=c_pred[i-t])
			if expl_con is not None and 'lin' in expl_con:
				boundary_x = np.linspace(self.plot_lims[0][0], self.plot_lims[0][1], 50)
				# for i in range(t, t+pred_len):
				for i in range(t+pred_len-1, t-1, -1): # Go backwards so that earlier stuff is plotted on top
					plot_t = min(i, agent_prev_cl.shape[1]-1)
					H = lin_con[0][plot_t]
					g = lin_con[1][plot_t]
					for j in range(H.shape[0]):
						boundary_y = (-H[j,0]*boundary_x - g[j])/(H[j,1]+1e-10)
						self.pos_ax.plot(boundary_x, boundary_y, '--', linewidth=0.7, c=c_pred[i-t])

					if shade:
						for j in range(P1.shape[0]):
							for k in range(P2.shape[1]):
								test_pt = np.array([P1[j,k], P2[j,k]])
								if np.all(H.dot(test_pt) + g <= 0):
									self.pos_ax.plot(test_pt[0], test_pt[1], '.', c=c_pred[i-t], markersize=1)

		# Plot the closed loop position trajectory up to this iteration and the optimal solution at this iteration

		self.pos_ax.scatter(pos_preds[0,:], pos_preds[1,:], marker='.', c=c_pred)
		self.pos_ax.plot(pos_cl[0,:], pos_cl[1,:], 'k.')

		# Plot the closed loop state trajectory up to this iteration and the optimal solution at this iteration
		for (i, a) in enumerate(self.state_axs):
			if self.prev_state_cl is not None:
				l = self.prev_state_cl[self.agent_id].shape[1]
				a.plot(range(l), self.prev_state_cl[self.agent_id][i,:], 'b.')
			a.plot(range(t, t+pred_len), state_preds[i,:], 'g.')
			a.plot(range(t, t+pred_len), state_preds[i,:], 'g')
			a.plot(range(cl_len), state_cl[i,:], 'k.')

		try:
			self.pos_fig.canvas.draw()
			self.state_fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

		# Save plots if plot_dir was specified
		if self.plot_dir is not None:
			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('pos', f_name))
			self.pos_fig.savefig('/'.join((self.plot_dir, f_name)))

			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('state', f_name))
			self.state_fig.savefig('/'.join((self.plot_dir, f_name)))

	def plot_act_traj(self, act_cl, act_preds, t):
		self.clear_act_plot()

		cl_len = act_cl.shape[1]
		pred_len = act_preds.shape[1]

		# Plot the closed loop input trajectory up to this iteration and the optimal solution at this iteration
		for (i, a) in enumerate(self.act_axs):
			if self.prev_act_cl is not None:
				l = self.prev_act_cl[self.agent_id].shape[1]
				a.plot(range(l), self.prev_act_cl[self.agent_id][i,:], 'b.')
			a.plot(range(t, t+pred_len), act_preds[i,:], 'g.')
			a.plot(range(t, t+pred_len), act_preds[i,:], 'g')
			a.plot(range(cl_len), act_cl[i,:], 'k.')

		try:
			self.act_fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

		if self.plot_dir is not None:
			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('act', f_name))
			self.act_fig.savefig('/'.join((self.plot_dir, f_name)))
