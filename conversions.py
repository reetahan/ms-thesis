from graphs import * 
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

class Conversion:
	def __init__(model,parties,conv_thresh, n, p_r, rho_col, p_b=-1,rho_excol=-1,rho_res=-1, seed_ct=30,iter_max=5):
		self.G = self.generate_graph(model,parties,n,p_r,p_b,rho_col,rho_excol,rho_res)
		self.party_count = parties
		self.theta = conv_thresh
		self.seeds = self.generate_seeds(seed_ct)

	def generate_graph(self,model,parties,n,p_r,p_b,rho_col,rho_excol,rho_res):
		if(model == 'BPA'):
			if(parties == 2):
				return BPA2PGraph(n,p_r,rho_col)
			if(parties == 3):
				return BPA3PGraph(n,p_r,p_b,rho_col,rho_res)
		if(model == 'SB'):
			if(parties == 2):
				return SB2PGraph(n,p_r,rho_col,rho_res)
			if(parties == 3):
				return SB3PGraph(n,p_r,p_b,rho_col,rho_excol,rho_res)


	def generate_seeds(self,seed_ct):
		return np.random.randint(self.seed_ct*1000,size=seed_ct)
	
	def rd3(num):
		return round(num,3)

	def color_two(self,node,red,blue):
		if(node in red):
			return "red"
		if(node in blue):
			return "blue"

	def color_three(self,node,red,blue,green):
		if(node in red):
			return "red"
		if(node in blue):
			return "blue"
		if(node in green):
			return "green"
	
	def run_simultaneous_conversions_two(self):

		G = self.G 
		red = G.red
		blue = G.blue 
		color_map = G.color_map
		conv_thresh = self.theta
		
		color_map_orig = copy.deepcopy(color_map)
		red_orig = copy.deepcopy(red)
		blue_orig = copy.deepcopy(blue)

		for n in G.nodes:

			neighbors = list(G.neighbors(n))
			my_color = self.color_two(n,red_orig,blue_orig)

			if(my_color == 'red'):
				my_color = red
			else:
				my_color = blue
			neighbor_colors = np.array([self.color_two(v,red_orig,blue_orig) for v in neighbors])
			t = conv_thresh * len(neighbor_colors)

			passing_color = []
			passing_values = []
			for c in ["red","blue"]:
				color_ct = np.count_nonzero(neighbor_colors == c)
				if(color_ct >= t):
					passing_color.append(c)
					passing_values.append(color_ct)

			selection = None
			if(len(passing_color) > 0):
				probs = [x/sum(passing_values) for x in passing_values]
				selection = np.random.choice(passing_color, p=probs)

			if(selection == 'red'):
				my_color.remove(n)
				color_map[n] = 'red'
				red.add(n)

			if(selection == 'blue'):
				my_color.remove(n)
				color_map[n] = 'blue'
				blue.add(n)

		self.G = G 
		self.red = red 
		self.blue = blue 
		self.color_map = color_map

	def run_simultaneous_conversions_three(G,red,blue,green,conv_thresh, color_map):
		G = self.G 
		red = G.red
		blue = G.blue 
		green = G.green
		color_map = G.color_map
		conv_thresh = self.theta
		
		color_map_orig = copy.deepcopy(color_map)
		red_orig = copy.deepcopy(red)
		blue_orig = copy.deepcopy(blue)
		green_orig = copy.deepcopy(green)

		for n in G.nodes:

			neighbors = G.neighbors(n)
			my_color = self.color_three(n,red,blue,green)
			cur_color = my_color

			if(my_color == 'red'):
				my_color = red
			elif(my_color == 'blue'):
				my_color = blue
			else:
				my_color = green
			neighbor_colors = np.array([self.color_three(v,red_orig,blue_orig,green_orig) for v in neighbors])
			t = conv_thresh * len(neighbor_colors)
		   

			passing_color = []
			passing_values = []
			for c in ["red","blue", "green"]:
				color_ct = np.count_nonzero(neighbor_colors == c)
				if(color_ct >= t):
					passing_color.append(c)
					passing_values.append(color_ct)

			selection = None
			if(len(passing_color) > 0):
				probs = [x/sum(passing_values) for x in passing_values]
				selection = np.random.choice(passing_color, p=probs)


			if(selection == 'red'):
				my_color.remove(n)
				color_map[n] = 'red'
				red.add(n)

			if(selection == 'blue'):
				my_color.remove(n)
				color_map[n] = 'blue'
				blue.add(n)
			if(selection == 'green'):
				my_color.remove(n)
				color_map[n] = 'green'
				green.add(n)

		self.G = G 
		self.red = red 
		self.blue = blue 
		self.green = green
		self.color_map = color_map

	def driver_simultaneous_conversions_two(self):
		red_avg = []
		blue_avg = []

		for seed in self.seeds:

			np.random.seed(seed)
				
			cur_reds = []
			cur_blues = []

			init_red = rd3(len(self.red)/len(self.G.nodes)*100)
			init_blue = rd3(len(self.blue)/len(self.G.nodes)*100)

			cur_reds.append(init_red)
			cur_blues.append(init_blue)

			
			iters = 0
			while(iters < self.iter_max):
				self.run_simultaneous_conversions_two()
				iters = iters + 1
				cur_reds.append(rd3(len(self.red)/len(self.G.nodes)*100))
				cur_blues.append(rd3(len(self.blue)/len(self.G.nodes)*100))
			
			print(seed, flush=True)
			
			red_avg.append(cur_reds[len(cur_reds)-1])
			blue_avg.append(cur_blues[len(cur_blues)-1])

		return rd3(np.average(red_avg)), rd3(np.average(blue_avg))

	def driver_simultaneous_conversions_three(self):

		red_avg = []
		blue_avg = []
		green_avg = []

		for seed in self.seeds:

			np.random.seed(seed)
				
			cur_reds = []
			cur_blues = []
			cur_greens = []

			init_red = rd3(len(self.red)/len(self.G.nodes)*100)
			init_blue = rd3(len(self.blue)/len(self.G.nodes)*100)
			init_green = rd3(len(self.green)/len(self.G.nodes)*100)

			cur_reds.append(init_red)
			cur_blues.append(init_blue)
			cur_greens.append(init_green)

			
			iters = 0
			while(iters < self.iter_max):
				self.run_simultaneous_conversions_three()
				iters = iters + 1
				cur_reds.append(rd3(len(self.red)/len(self.G.nodes)*100))
				cur_blues.append(rd3(len(self.blue)/len(self.G.nodes)*100))
				cur_greens.append(rd3(len(self.green)/len(self.G.nodes)*100))
			
			print(seed, flush=True)
			
			red_avg.append(cur_reds[len(cur_reds)-1])
			blue_avg.append(cur_blues[len(cur_blues)-1])
			green_avg.append(cur_greens[len(cur_greens)-1])

		return rd3(np.average(red_avg)), rd3(np.average(blue_avg)), rd3(np.average(green_avg))

	def run_simultaneous_conversions(self):
		if(self.party_count == 2):
			return self.driver_simultaneous_conversions_two()
		if(self.party_count == 3):
			return self.driver_simultaneous_conversions_three()