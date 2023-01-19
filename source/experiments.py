from graphs import *
from conversions import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def rd3(num):
	return round(num,3)

class Experiment:

	def __init__(self,model, party_count, fname_prefix, test_variable_name, test_variable_range, plot_xlabel, constants,
		plot_ylabel='Final Proportions of Parties', n=5000, iter_max=5, seed_ct=30):

		self.model = model
		self.party_count = party_count
		self.fname_prefix = fname_prefix
		self.var = test_variable_name
		self.range = test_variable_range
		self.xlabel = plot_xlabel
		self.constants = constants
		self.ylabel = plot_ylabel
		self.n = n
		self.iter_max = iter_max
		self.seed_ct = seed_ct
		self.fnames = []
		self.titles = []
	
	def run_2bpa(self):

		list_of_others = []
		cores_map = []

		for const in self.constants:
			list_of_others.append(self.constants[const])
			cores_map.append(const)
		total_tuples = []
		for a in list_of_others[0]:
			for b in list_of_others[1]:
				total_tuples.append((a, b))

		rho_col_idx = None
		p_r_idx = None
		conv_thresh_idx = None

		for i in range(len(cores_map)):
			if(cores_map[i] == "rho_col"):
				rho_col_idx = i
			if(cores_map[i] == "init_ratio"):
				p_r_idx = i
			if(cores_map[i] == "conv_thresh"):
				conv_thresh_idx = i


		for cur_tup in total_tuples:
			print("Current Tuple: " + str(cur_tup), flush=True)
			if(rho_col_idx != None):
				rho_col = cur_tup[rho_col_idx]
			if(p_r_idx != None):
				p_r = cur_tup[p_r_idx][0]
				p_b = cur_tup[p_r_idx][1]
			if(conv_thresh_idx != None):
				conv_thresh = cur_tup[conv_thresh_idx]

			if(rho_col_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(
					rd3(p_b)) + ", conv_thresh: " + str(rd3(conv_thresh))
				fname = self.fname_prefix + "_RHOCOL_" + "_prb_" + \
						str((rd3(p_r), rd3(p_b))) + "_convthresh_" + \
							str(rd3(conv_thresh)) + ".txt"
			if(conv_thresh_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + \
										", p_b: " + \
											str(rd3(p_b)) + ", rho_col: " + \
												str(rd3(rho_col))
				fname = self.fname_prefix + "_CONVTHRESH_" + "rc_" + \
						str(rd3(rho_col)) + "_prb_" + \
							str((rd3(p_r), rd3(p_b))) + ".txt"
			self.fnames.append(fname)
			self.titles.append(title_str)			

			init_diffs = []
			red_p = []
			blue_p = []
			for var_choice in self.range:
				if(rho_col_idx == None):
					rho_col = var_choice
					init_str = title_str + ", rho_col:" + str(rd3(rho_col))
					init_diffs.append(100*(rd3(rho_col)))
					
				if(conv_thresh_idx == None):
					conv_thresh = var_choice
					init_str = title_str + ", conv_thresh:" + \
						str(rd3(conv_thresh))
					init_diffs.append(100*(rd3(conv_thresh)))
					

				print(init_str, flush=True)

				conv = Conversion(model="BPA", parties=2, conv_thresh=conv_thresh,
					n=self.n, p_r=p_r, rho_col=rho_col, seed_ct=self.seed_ct, iter_max=self.iter_max)
				red_res, blue_res = conv.run_simultaneous_conversions()
				red_p.append(red_res)
				blue_p.append(blue_res)

			self.outfile_write_two(
				fname, title_str, self.xlabel, init_diffs, red_p, blue_p)


	def outfile_write_two(self,fname,title,xlabel,x_axis,red,blue):
		f = open(fname, "w+")
		f.write(title + "\n")
		f.write(xlabel + "\n")
		for x in x_axis:
			f.write(str(x) + "\n")
		f.write("red_p\n")
		for x in red:
			f.write(str(x) + "\n")
		f.write("blue_p\n")
		for x in blue:
			f.write(str(x) + "\n")
		f.close()

	def outfile_write_three(self,fname,title,xlabel,x_axis,red,blue,green):
		f = open(fname, "w+")
		f.write(title + "\n")
		f.write(xlabel + "\n")
		for x in x_axis:
			f.write(str(x) + "\n")
		f.write("red_p\n")
		for x in red:
			f.write(str(x) + "\n")
		f.write("blue_p\n")
		for x in blue:
			f.write(str(x) + "\n")
		f.write("green_p\n")
		for x in green:
			f.write(str(x) + "\n")
		f.close()
	
	def run_2sb(self):

		list_of_others = []
		cores_map = []

		for const in self.constants:
			list_of_others.append(self.constants[const])
			cores_map.append(const)
		total_tuples = []
		for a in list_of_others[0]:
			for b in list_of_others[1]:
				for c in list_of_others[2]:
					total_tuples.append((a,b,c))

		rho_col_idx = None
		rho_res_idx = None 
		p_r_idx = None 
		conv_thresh_idx = None

		for i in range(len(cores_map)):
			if(cores_map[i] == "rho_col"):
				rho_col_idx = i 
			if(cores_map[i] == "rho_res"):
				rho_res_idx = i 
			if(cores_map[i] == "init_ratio"):
				p_r_idx = i 
			if(cores_map[i] == "conv_thresh"):
				conv_thresh_idx = i 

		for cur_tup in total_tuples:
			print("Current Tuple: " +  str(cur_tup), flush=True)
			if(rho_col_idx != None):
				rho_col = cur_tup[rho_col_idx]
			if(rho_res_idx != None):
				rho_res = cur_tup[rho_res_idx]
			if(p_r_idx != None):
				p_r = cur_tup[p_r_idx][0]
				p_b = cur_tup[p_r_idx][1]
			if(conv_thresh_idx != None):
				conv_thresh = cur_tup[conv_thresh_idx]


			if(rho_col_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b))  + ", rho_res: " + str(rd3(rho_res)) + ", conv_thresh: " + str(rd3(conv_thresh))
				fname = self.fname_prefix + "_RHOCOL_" + "_prb_"  + str((rd3(p_r),rd3(p_b))) + "_rhores_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"  
			if(rho_res_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", rho_col: " + str(rd3(rho_col))  + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHORES_" + "_prb_"  + str((rd3(p_r),rd3(p_b))) + "_rhocol_" + str(rd3(rho_col)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(conv_thresh_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res)) 
				fname = self.fname_prefix + "_CONVTHRESH_"  + "_prb_" + str((rd3(p_r),rd3(p_b))) + "_rhocol_" + str(rd3(rho_col)) + "_rhores_" + str(rd3(rho_res)) + ".txt"
			self.fnames.append(fname)
			self.titles.append(title_str)

			init_diffs = []
			red_p = []
			blue_p = []
	
			for var_choice in self.range:
				if(rho_col_idx == None):
					rho_col = var_choice
					init_str = title_str + ", rho_col:" + str(rd3(rho_col)) 
					init_diffs.append(100*(rd3(rho_col)))
					
				if(rho_res_idx == None):
					rho_res = var_choice
					init_str = title_str + ", rho_res:" + str(rd3(rho_res)) 
					init_diffs.append(100*(rd3(rho_res)))
					
				if(conv_thresh_idx == None):
					conv_thresh = var_choice
					init_str = title_str + ", conv_thresh:" + str(rd3(conv_thresh))
					init_diffs.append(100*(rd3(conv_thresh))) 
				
				print(init_str, flush=True)

				conv = Conversion(model="SB",parties=2,conv_thresh=conv_thresh, 
					n=self.n, p_r=p_r, rho_col=rho_col, rho_res = rho_res,seed_ct=self.seed_ct,iter_max=self.iter_max)
				red_res, blue_res = conv.run_simultaneous_conversions()
				red_p.append(red_res)
				blue_p.append(blue_res)

			self.outfile_write_two(fname,title_str,self.xlabel,init_diffs,red_p,blue_p)

	def run_3bpa(self):

		list_of_others = []
		cores_map = []

		for const in self.constants:
			list_of_others.append(self.constants[const])
			cores_map.append(const)
		total_tuples = []
		for a in list_of_others[0]:
			for b in list_of_others[1]:
				for c in list_of_others[2]:
					total_tuples.append((a,b,c))

		rho_col_idx = None 
		rho_res_idx = None 
		p_r_idx = None 
		conv_thresh_idx = None

		for i in range(len(cores_map)):
			if(cores_map[i] == "rho_col"):
				rho_col_idx = i 
			if(cores_map[i] == "rho_res"):
				rho_res_idx = i 
			if(cores_map[i] == "init_ratio"):
				p_r_idx = i 
			if(cores_map[i] == "conv_thresh"):
				conv_thresh_idx = i 

		for cur_tup in total_tuples:
			print("Current Tuple: " +  str(cur_tup), flush=True)
			if(rho_col_idx != None):
				rho_col = cur_tup[rho_col_idx]
			if(rho_res_idx != None):
				rho_res = cur_tup[rho_res_idx]
			if(p_r_idx != None):
				p_r = cur_tup[p_r_idx][0]
				p_b = cur_tup[p_r_idx][1]
				p_g = cur_tup[p_r_idx][2]
			if(conv_thresh_idx != None):
				conv_thresh = cur_tup[conv_thresh_idx]

			if(rho_col_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))  + ", rho_res: " + str(rd3(rho_res)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHOCOL_" + "_prbg_"  + str((rd3(p_r),rd3(p_b),rd3(p_g)))+ "_rhores_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(rho_res_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))  + ", rho_col: " + str(rd3(rho_col)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHORES_" + "_prbg_"  + str((rd3(p_r),rd3(p_b),rd3(p_g)))+ "_rhocol_" + str(rd3(rho_col)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(conv_thresh_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))  + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res)) 
				fname = self.fname_prefix + "_CONVTHRESH_" + "_prbg_" + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_rhocol_" + str(rd3(rho_col)) + "_rhores_" + str(rd3(rho_res)) + ".txt"
			self.fnames.append(fname)
			self.titles.append(title_str)	

			init_diffs = []
			red_p = []
			blue_p = []
			green_p = []
	
			for var_choice in self.range:
				if(rho_col_idx == None):
					rho_col = var_choice
					init_str = title_str + ", rho_col:" + str(rd3(rho_col)) 
					init_diffs.append(100*(rd3(rho_col)))
				if(rho_res_idx == None):
					rho_res = var_choice
					init_str = title_str + ", rho_res:" + str(rd3(rho_res)) 
					init_diffs.append(100*(rd3(rho_col)))	
				if(conv_thresh_idx == None):
					conv_thresh = var_choice
					init_str = title_str + ", conv_thresh:" + str(rd3(conv_thresh))
					init_diffs.append(100*(rd3(conv_thresh))) 
				
				print(init_str, flush=True)

				conv = Conversion(model="BPA",parties=3,conv_thresh=conv_thresh, 
					n=self.n, p_r=p_r,p_b=p_b, rho_col=rho_col,rho_res=rho_res,seed_ct=self.seed_ct,iter_max=self.iter_max)
				red_res, blue_res, green_res = conv.run_simultaneous_conversions()
				red_p.append(red_res)
				blue_p.append(blue_res)

			self.outfile_write_three(fname,title_str,self.xlabel,init_diffs,red_p,blue_p,green_p)

	def run_3sb(self):

		list_of_others = []
		cores_map = []

		for const in self.constants:
			list_of_others.append(self.constants[const])
			cores_map.append(const)
		total_tuples = []
		for a in list_of_others[0]:
			for b in list_of_others[1]:
				for c in list_of_others[2]:
					for d in list_of_others[3]:
						total_tuples.append((a,b,c,d))

		rho_col_idx = None 
		rho_excol_idx = None 
		rho_res_idx = None 
		p_r_idx = None 
		conv_thresh_idx = None

		for i in range(len(cores_map)):
			if(cores_map[i] == "rho_col"):
				rho_col_idx = i 
			if(cores_map[i] == "rho_excol"):
				rho_excol_idx = i 
			if(cores_map[i] == "rho_res"):
				rho_res_idx = i 
			if(cores_map[i] == "init_ratio"):
				p_r_idx = i 
			if(cores_map[i] == "conv_thresh"):
				conv_thresh_idx = i 

		for cur_tup in total_tuples:
			print("Current Tuple: " +  str(cur_tup), flush=True)
			if(rho_col_idx != None):
				rho_col = cur_tup[rho_col_idx]
			if(rho_excol_idx != None):
				rho_excol = cur_tup[rho_excol_idx]
			if(rho_res_idx != None):
				rho_res = cur_tup[rho_res_idx]
			if(p_r_idx != None):
				p_r = cur_tup[p_r_idx][0]
				p_b = cur_tup[p_r_idx][1]
				p_g = cur_tup[p_r_idx][2]
			if(conv_thresh_idx != None):
				conv_thresh = cur_tup[conv_thresh_idx]


			if(rho_col_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))+ ", rho_excol: " + str(rd3(rho_excol))  + ", rho_res: " + str(rd3(rho_res)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHOCOL_" + "_prbg_"  + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_rhoexcol_" + str(rd3(rho_excol)) + "_rhores_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(rho_excol_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))+ ", rho_col: " + str(rd3(rho_excol))  + ", rho_res: " + str(rd3(rho_res)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHOEXCOL_" + "_prbg_"  + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_rhocol_" + str(rd3(rho_col)) + "_rhores_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(rho_res_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))  + ", rho_col: " + str(rd3(rho_col))+ ", rho_excol: " + str(rd3(rho_excol)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
				fname = self.fname_prefix + "_RHORES_" + "_prbg_"  + str((rd3(p_r),rd3(p_b),rd3(p_g)))+ "_rhocol_" + str(rd3(rho_col))  + "_rhoexcol_" + str(rd3(rho_excol)) + "_convthresh_" + str(rd3(conv_thresh)) + ".txt"
			if(conv_thresh_idx == None):
				title_str = "N: " + str(self.n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g))  + ", rho_col: " + str(rd3(rho_col))+ ", rho_excol: " + str(rd3(rho_excol)) + ", rho_res: " + str(rd3(rho_res)) 
				fname = self.fname_prefix + "_CONVTHRESH_" + "_prbg_" + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_rhocol_" + str(rd3(rho_col))  + "_rhoexcol_" + str(rd3(rho_excol)) + "_rhores_" + str(rd3(rho_res)) + ".txt"
			self.fnames.append(fname)
			self.titles.append(title_str)	

			init_diffs = []
			red_p = []
			blue_p = []
			green_p = []
	
			for var_choice in self.range:
				if(rho_col_idx == None):
					rho_col = var_choice
					init_str = title_str + ", rho_col:" + str(rd3(rho_col)) 
					init_diffs.append(100*(rd3(rho_col)))		
				if(rho_excol_idx == None):
					rho_excol = var_choice
					init_str = title_str + ", rho_excol:" + str(rd3(rhoex_col)) 
					init_diffs.append(100*(rd3(rho_excol)))		
				if(rho_res_idx == None):
					rho_res = var_choice
					init_str = title_str + ", rho_res:" + str(rd3(rho_res)) 
					init_diffs.append(100*(rd3(rho_col)))
				if(conv_thresh_idx == None):
					conv_thresh = var_choice
					init_str = title_str + ", conv_thresh:" + str(rd3(conv_thresh))
					init_diffs.append(100*(rd3(conv_thresh))) 
					
				print(init_str, flush=True)

				conv = Conversion(model="SB",parties=3,conv_thresh=conv_thresh, 
					n=self.n, p_r=p_r, p_b=p_b, rho_col=rho_col, rho_excol=rho_excol,
					rho_res=rho_res,seed_ct=self.seed_ct,iter_max=self.iter_max)
				red_res, blue_res, green_res = conv.run_simultaneous_conversions()
				red_p.append(red_res)
				blue_p.append(blue_res)
				green_p.append(green_res)

			self.outfile_write_three(fname,title_str,self.xlabel,init_diffs,red_p,blue_p,green_p)

	def run_experiment(self):
		if(self.party_count == 3):
			if(self.model == 'BPA'):
				self.run_3bpa()
			if(self.model == 'SB'):
				self.run_3sb()
		if(self.party_count == 2):
			if(self.model == 'BPA'):
				self.run_2bpa()
			if(self.model == 'SB'):
				self.run_2sb()

		
