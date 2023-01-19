from graphs import *
from conversions import *
from experiments import *
from plots import *
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import copy


def main():

	# 2-party BPA
	info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"}} 
	other_info = {"conv_thresh": [0.35,0.55,0.75], "rho_col": [0.3,0.5,0.7], "init_ratio":[(0.55,0.45),(0.8,0.2)]}

	fname_prefix='2p_BPA_results/NEW_2P_BPA_simultaneous'
	y_lab = 'Final Proportions of Parties'
	model = 'BPA'
	party_count = 2
	N = 5000
	iter_max = 5
	seed_ct = 30

	'''
	# 2-party SB
	info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"},"rho_res":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho res (0% to 100%)"}} 
	other_info = {"conv_thresh": [0.35,0.55,0.75], "rho_col": [0.4,0.6,0.8],"rho_res": [0.1,0.3,0.5], "init_ratio":[(0.55,0.45),(0.8,0.2)]}

	fname_prefix='2p_SB_results/NEW_2P_SB_simultaneous'
	y_lab = 'Final Proportions of Parties'
	model = 'SB'
	party_count = 2
	N = 5000
	iter_max = 5
	seed_ct = 30
	'''

	'''
	# 3-party BPA
	info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"},"rho_res":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho res (0% to 100%)"}} 
	other_info = {"conv_thresh": [0.35,0.55,0.75], "rho_col": [0.4,0.6,0.8], "rho_res": [0.1,0.3,0.5], "init_ratio":[(0.55,0.15,0.3),(0.3,0.15,0.55),(0.4,0.2,0.4),(0.4,0.4,0.2),(0.35,0.25,0.4),(0.4,0.25,0.35)]}

	fname_prefix='3p_BPA_results/NEW_3P_BPA_simultaneous'
	y_lab = 'Final Proportions of Parties'
	model = 'BPA'
	party_count = 3
	N = 5000
	iter_max = 5
	seed_ct = 30
	'''

	'''
	# 3-party SB
	info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"}, "rho_excol":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho excol (0% to 100%)"}, "rho_res":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho res (0% to 100%)"}} 
	other_info = {"conv_thresh": [0.35,0.55,0.75], "rho_col": [0.4,0.6,0.8],"rho_excol": [0.25,0.45,0.65], "rho_res": [0.1,0.3,0.5], "init_ratio":[(0.55,0.15,0.3),(0.3,0.15,0.55),(0.4,0.2,0.4),(0.4,0.4,0.2),(0.35,0.25,0.4),(0.4,0.25,0.35)]}

	fname_prefix='3p_SB_results/NEW_3P_SB_simultaneous'
	y_lab = 'Final Proportions of Parties'
	model = 'SB'
	party_count = 3
	N = 5000
	iter_max = 5
	seed_ct = 30
	'''

	for question in info:

		exp_consts = {}
		for q in other_info:
			if(q != question):
				exp_consts[q] = other_info[q]

		exp = Experiment(model=model,party_count=party_count,fname_prefix=fname_prefix,test_variable_name=question,
			test_variable_range=info[question]['vals'], plot_xlabel=info[question]['xlabel'], constants=exp_consts, plot_ylabel=y_lab,
			n=N,iter_max=iter_max,seed_ct=seed_ct)
		exp.run_experiment()

		p = Plot(fnames=exp.fnames, titles=exp.titles,party_count=2,var_range=info[question]['vals'],xlabel=info[question]['xlabel'],ylabel=y_lab)
		p.generate_plot(show=True)
	

if __name__ == '__main__':
	main()