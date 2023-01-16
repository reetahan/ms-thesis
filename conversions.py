from graphs import * 
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

class Conversion:
	def __init__(graph,conv_thresh,seed_ct=30):
		self.G = graph
		self.theta = conv_thresh
		self.seed_ct = seed_ct

		self.seeds = self.generate_seeds()

	def generate_seeds(self):
		return np.random.randint(self.seed_ct*1000,size=seed_ct)

	def run_simultaneous_conversions(self):
		if(self.party_count == 2):
			return self.run_simultaneous_conversions_two()
		if(self.party_count == 3):
			return self.run_simultaneous_conversions_three()

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
	        my_color = color(n,red_orig,blue_orig)

	        if(my_color == 'red'):
	            my_color = red
	        else:
	            my_color = blue
	        neighbor_colors = np.array([color(v,red_orig,blue_orig) for v in neighbors])
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
	    return G,color_map,red,blue

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
	        my_color = color(n,red,blue,green)
	        cur_color = my_color

	        if(my_color == 'red'):
	            my_color = red
	        elif(my_color == 'blue'):
	            my_color = blue
	        else:
	            my_color = green
	        neighbor_colors = np.array([color(v,red_orig,blue_orig,green_orig) for v in neighbors])
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
	    return G,color_map,red,blue,green
