import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import copy, pickle, pdb, argparse, os, sys

BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-1]))
sys.path.append(BASE_DIR)

import utils.plot_utils
import utils.utils

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str, help='Experiment to plot')
parser.add_argument('-v', '--video', action='store_true', help='Flag for creating and saving videos')

args = parser.parse_args()

def main():
	out_dir = '/'.join((BASE_DIR, 'out'))
	exp_dir = '/'.join((out_dir, args.exp))

	plot_lims = [[-2.5, 2.5], [-2.5, 2.5]]
	r_a = [0.1, 0.2, 0.3] # Agents are circles with radius r_a

	max_it = np.amax([int(n.split('_')[-1]) for n in os.listdir(exp_dir)
		if os.path.isdir(os.path.join(exp_dir, n))])

	# Generate videos for each iteration
	if args.video:
		print('Generating videos')

		iters = range(0, max_it+1)

		f = open('/'.join((exp_dir, 'it_%i.pkl' % max_it)), 'r')
		lmpc = pickle.load(f)
		for i in iters:
			xcl = []
			for l in lmpc:
				xcl.append(l.SS[i])
			f, f_a = utils.plot_utils.plot_agent_trajs(xcl, r_a=r_a, plot_lims=plot_lims, save_dir=exp_dir, save_video=True, it=i)
			plt.close(f)
			plt.close(f_a)

	# Plot iteration cost
	f = open('/'.join((exp_dir, 'it_%i.pkl' % max_it)), 'r')
	lmpc = pickle.load(f)

	fig_cost = plt.figure()
	symbols = ['s', 'd', 'o']
	for (i, l) in enumerate(lmpc):
		cost = []
		for q in l.Qfun:
			cost.append(q[0])
		plt.plot(range(len(cost)), cost, '-'+symbols[i], label=('Agent %i' % (i+1)))
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Cost')
	fig_cost.savefig('/'.join((exp_dir, 'cost.png')))

	plt.show()

if __name__== "__main__":
  main()
