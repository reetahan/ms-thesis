from graphs import *
from conversions import *
from experiments import *
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import copy


def main():

	info = {"conv_thresh": {"vals": np.arange(0.00, 1.01, 0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col": {
											  "vals": np.arange(0.00, 1.01, 0.05), "xlabel": "Rho col (0% to 100%)"}}
											  
	other_info = {"conv_thresh": [0.35,0.55,0.75], "rho_col": [0.3,0.5,0.7], "init_ratio":[(0.55,0.45),(0.8,0.2)]}
	y_lab = 'Final Proportions of Parties'
	N = 5000
	iter_max = 5

	for question in info:

		exp_consts = {}
		for q in other_info:
			if(q != question):
				exp_consts[q] = other_info[q]

		exp = Experiment(model='BPA',party_count=2,fname_prefix='2p_BPA_results/TEST_2P_BPA_simultaneous',test_variable_name=question,
			test_variable_range=info[question]['vals'], plot_xlabel=info[question]['xlabel'], constants=exp_consts, plot_ylabel=y_lab,
			n=N,iter_max=iter_max,seed_ct=30)
		exp.run_experiment()
	

if __name__ == '__main__':
	main()