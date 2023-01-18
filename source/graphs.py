import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Graph:
	pass


class BPA2PGraph(Graph):
	def __init__(self,n, p_r, rho_col):
		self.n = n
		self.party_count = 2
		self.p_r = p_r
		self.p_b = 1.0 - self.p_r
		self.rho_col = rho_col

		self.graph, self.red, self.blue, self.color_map = self.generate_graph()

	def generate_graph(self):
		n = self.n
		p_r = self.p_r
		p_b = self.p_b
		rho_col = self.rho_col

		num_red = n * p_r
		num_blue = n * p_b

		red = set({0})
		blue = set({1})

		G = nx.MultiGraph()
		G.add_edge(0,1)

		degree_dict = {0:1,1:1}
		
		color_map = ['red','blue']


		for i in range(2, n):

			red_ind = False
			blue_ind = False

			selection = np.random.choice(["red","blue"],p=[p_r,p_b])
	 
			if(selection == "red"):
				red_ind = True
				color_map.append('red')
				red.add(i)
			if(selection == "blue"):
				blue_ind = True
				color_map.append('blue')
				blue.add(i)
			G.add_node(i)

			added = False
			while(added == False):
				
				exist_node_degrees = np.array([*degree_dict.values()])
				exist_node_degrees_sum = np.sum(exist_node_degrees) 
				exist_node_degrees = exist_node_degrees/exist_node_degrees_sum
				exist_node_keys = [*degree_dict.keys()]
				exist_node = np.random.choice(exist_node_keys,p=exist_node_degrees)

				if(red_ind):
					if exist_node in red:
						G.add_edge(i, exist_node)
						added = True
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						continue
					edge_ind = np.random.binomial(1,rho_col)
					if(edge_ind):
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True

				if(blue_ind):
					if exist_node in blue:
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True
						continue 
					edge_ind = np.random.binomial(1,rho_col)
					if(edge_ind):
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True


		return G,red,blue, color_map

	def graph_analysis(self):
		G = self.graph 
		red = self.red 
		blue = self.blue 
		color_map = self.color_map

		deg_dist = []
		red_dist = []
		blue_dist = []
		rf_dist = []
		for n in G.nodes():
			nb = list(G.neighbors(n))
			deg_dist.append(len(nb))
			if(n in red):
				red_dist.append(len(nb))
			else:
				blue_dist.append(len(nb))
			ct = 0
			for v in nb:
				if(self.color(v) == 'red'):
					ct += 1
			rf_dist.append(ct/len(nb))

		plt.hist(deg_dist, bins=20, color='green')
		plt.title("Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(red_dist, bins=20, color='red')
		plt.title("Red Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(blue_dist, bins=20)
		plt.title("Blue Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(rf_dist, bins=20, color='red')
		plt.title("Red Degree Fraction Distribution")
		plt.show()
		plt.clf()


class SB2PGraph(Graph):
	def __init__(self,n,p_r,rho_col, rho_res):
		self.n = n 
		self.p_r = p_r
		self.party_count = 2
		self.p_b = 1.0 - self.p_r 
		self.rho_col = rho_col
		self.rho_res = rho_res

		self.graph, self.red, self.blue, self.color_map = self.generate_graph()

	def generate_graph(self):
		n = self.n 
		p_r = self.p_r
		p_b = self.p_b
		rho_col = self.rho_col
		rho_res = self.rho_res

		print('Start graph of size ' + str(n) + ", p_r = " + str(p_r) + ", rho_col = " + str(rho_col) + ", rho_res = " + str(rho_res))
		num_red = n * p_r
		num_blue = n - num_red

		G = nx.MultiGraph()
		red = set()
		blue = set()
		color_map = []

		for i in range(n):

			if(i < num_red):
				red.add(i)
				color_map.append('red')
				G.add_node(i)
			else:
				blue.add(i)
				color_map.append('blue')
				G.add_node(i)


		for i in range(n):
			for j in range(i+1,n):
				if((i < num_red and j < num_red) or (i >= num_red and j >= num_red)):
					edge_ind = np.random.binomial(1,rho_col)
				if((i < num_red and j >= num_red) or (i >= num_red and j < num_red)):
					edge_ind = np.random.binomial(1,rho_res)
				if(edge_ind):
					G.add_edge(i,j)

		print('Graph created!')
		return G,red,blue,color_map

	def color(self,node):
		if(node in self.red):
			return "red"
		if(node in self.blue):
			return "blue"

	def graph_analysis(self):
		G = self.graph 
		red = self.red 
		blue = self.blue 
		color_map = self.color_map

		deg_dist = []
		red_dist = []
		blue_dist = []
		rf_dist = []
		for n in G.nodes():
			nb = list(G.neighbors(n))
			deg_dist.append(len(nb))
			if(n in red):
				red_dist.append(len(nb))
			else:
				blue_dist.append(len(nb))
			ct = 0
			for v in nb:
				if(self.color(v) == 'red'):
					ct += 1
			rf_dist.append(ct/len(nb))

		plt.hist(deg_dist, bins=20, color='green')
		plt.title("Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(red_dist, bins=20, color='red')
		plt.title("Red Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(blue_dist, bins=20)
		plt.title("Blue Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(rf_dist, bins=20, color='red')
		plt.title("Red Degree Fraction Distribution")
		plt.show()
		plt.clf()


class BPA3PGraph(Graph):
	def __init__(self,n, p_r,p_b,rho_col, rho_res):
		self.n = n 
		self.p_r = p_r
		self.p_b = p_b 
		self.party_count = 3
		self.p_g = 1.0 - self.p_r - self.p_b
		self.rho_col = rho_col
		self.rho_res = rho_res

		self.graph, self.red, self.blue, self.green, self.color_map = self.generate_graph()

	def generate_graph(self):
		n = self.n 
		p_r = self.p_r
		p_b = self.p_b
		p_g = self.p_g
		rho_col = self.rho_col
		rho_res = self.rho_res

		num_red = n * p_r
		num_blue = n * p_b
		num_green =  n * p_g

		red = set({0})
		blue = set({1})
		green = set({2})

		G = nx.MultiGraph()
		G.add_edge(0,1)
		G.add_edge(0,2)
		G.add_edge(1,2)

		degree_dict = {0:2,1:2,2:2}
		
		color_map = ['red','blue','green']


		for i in range(3, n):

			red_ind = False
			blue_ind = False
			green_ind = False

			selection = np.random.choice(["red","blue","green"],p=[p_r,p_b,p_g])
	 
			if(selection == "red"):
				red_ind = True
				color_map.append('red')
				red.add(i)
			if(selection == "blue"):
				blue_ind = True
				color_map.append('blue')
				blue.add(i)
			if(selection == "green"):
				green_ind = True
				color_map.append('green')
				green.add(i)
			G.add_node(i)

			added = False
			while(added == False):

				exist_node_degrees = np.array([*degree_dict.values()])
				exist_node_degrees_sum = np.sum(exist_node_degrees) 
				exist_node_degrees = exist_node_degrees/exist_node_degrees_sum
				exist_node_keys = [*degree_dict.keys()]
				exist_node = np.random.choice(exist_node_keys,p=exist_node_degrees)
				
				if(red_ind):
					if exist_node in red:
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True
						continue
					edge_ind = 0
					if(exist_node in blue):
						edge_ind = np.random.binomial(1,rho_col)
					if(exist_node in green):
						edge_ind = np.random.binomial(1,rho_res)
					if(edge_ind):
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True

				if(blue_ind):
					if exist_node in blue:
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True
						continue
					edge_ind = 0
					if(exist_node in red):
						edge_ind = np.random.binomial(1,rho_col)
					if(exist_node in green):
						edge_ind = np.random.binomial(1,rho_res)
					if(edge_ind):
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True

				if(green_ind):
					if exist_node in green:
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True
						continue
					edge_ind = 0
					if(exist_node in blue or exist_node in red):
						edge_ind = np.random.binomial(1,rho_res)
					if(edge_ind):
						G.add_edge(i, exist_node)
						degree_dict[i] = 1
						degree_dict[exist_node] += 1
						added = True

		return G,red,blue,green, color_map

	def color(self,node):
		if(node in self.red):
			return "red"
		if(node in self.blue):
			return "blue"
		if(node in self.green):
			return "green"

	def graph_analysis(self):
		G = self.graph 
		red = self.red 
		blue = self.blue 
		green = self.green
		color_map = self.color_map

		deg_dist = []
		red_dist = []
		blue_dist = []
		green_dist = []
		rf_dist = []
		gf_dist = []
		for n in G.nodes():
			nb = list(G.neighbors(n))
			deg_dist.append(len(nb))
			if(n in red):
				red_dist.append(len(nb))
			else:
				if(n in green):
					green_dist.append(len(nb))
				else:
					blue_dist.append(len(nb))
			ct = 0
			ct_g = 0
			for v in nb:
				if(color(v) == 'red'):
					ct += 1
				if(color(v) == 'green'):
					ct_g += 1
			rf_dist.append(ct/len(nb))
			gf_dist.append(ct_g/len(nb))

		plt.hist(deg_dist, bins=20, color='green')
		plt.title("Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(red_dist, bins=20, color='red')
		plt.title("Red Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(blue_dist, bins=20)
		plt.title("Blue Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(rf_dist, bins=20, color='red')
		plt.title("Red Degree Fraction Distribution")
		plt.show()
		plt.clf()

		plt.hist(gf_dist, bins=20, color='red')
		plt.title("Green Degree Fraction Distribution")
		plt.show()
		plt.clf()


class SB3PGraph(Graph):
	def __init__(self,n,p_r, p_b,rho_col,rho_excol,rho_res):
		self.n = n 
		self.p_r = p_r
		self.p_b = p_b 
		self.party_count = 3
		self.p_g = 1.0 - self.p_r - self.p_b
		self.rho_col = rho_col
		self.rho_excol = rho_excol
		self.rho_res = rho_res

		self.graph, self.red, self.blue, self.green, self.color_map = self.generate_graph()

	def generate_graph(self):
		n = self.n 
		p_r = self.p_r
		p_b = self.p_b
		p_g = self.p_g
		rho_col = self.rho_col
		rho_excol = self.rho_excol
		rho_res = self.rho_res

		print('Start graph of size ' + str(n) + ", p_r = " + str(p_r) + ", rho_col = " + str(rho_col) + ", rho_excol = " + str(rho_excol) + ", rho_res = " + str(rho_res))
		num_red = n * p_r
		num_blue = n * p_b
		num_green = n - num_blue- num_red

		G = nx.MultiGraph()
		red = set()
		blue = set()
		green = set()
		color_map = []

		for i in range(n):

			if(i < num_red):
				red.add(i)
				color_map.append('red')
				G.add_node(i)
			else:
				if(i < num_red + num_blue):
					blue.add(i)
					color_map.append('blue')
					G.add_node(i)
				else:
					green.add(i)
					color_map.append('green')
					G.add_node(i)


		for i in range(n):
			for j in range(i+1,n):
				if((i < num_red and j < num_red) or (i >= num_red and i <num_blue and j >= num_red and j < num_blue) or (i > num_blue and j > num_blue)):
					edge_ind = np.random.binomial(1,rho_col)
				if((i < num_red and j >= num_red and j < num_blue)):
					edge_ind = np.random.binomial(1,rho_excol)
				if((i < num_red and j >= num_blue) or (i > num_red and i < num_blue and j>= num_blue)):
					edge_ind = np.random.binomial(1,rho_res)
				if(edge_ind):
					G.add_edge(i,j)

		print('Graph created!')
		return G, red, blue, green, color_map

	def color(self,node):
		if(node in self.red):
			return "red"
		if(node in self.blue):
			return "blue"
		if(node in self.green):
			return "green"

	def graph_analysis(self):
		G = self.graph 
		red = self.red 
		blue = self.blue 
		green = self.green
		color_map = self.color_map

		deg_dist = []
		red_dist = []
		blue_dist = []
		green_dist = []
		rf_dist = []
		gf_dist = []
		for n in G.nodes():
			nb = list(G.neighbors(n))
			deg_dist.append(len(nb))
			if(n in red):
				red_dist.append(len(nb))
			else:
				if(n in green):
					green_dist.append(len(nb))
				else:
					blue_dist.append(len(nb))
			ct = 0
			ct_g = 0
			for v in nb:
				if(color(v) == 'red'):
					ct += 1
				if(color(v) == 'green'):
					ct_g += 1
			rf_dist.append(ct/len(nb))
			gf_dist.append(ct_g/len(nb))

		plt.hist(deg_dist, bins=20, color='green')
		plt.title("Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(red_dist, bins=20, color='red')
		plt.title("Red Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(blue_dist, bins=20)
		plt.title("Blue Node Degree Distribution")
		plt.show()
		plt.clf()

		plt.hist(rf_dist, bins=20, color='red')
		plt.title("Red Degree Fraction Distribution")
		plt.show()
		plt.clf()

		plt.hist(gf_dist, bins=20, color='red')
		plt.title("Green Degree Fraction Distribution")
		plt.show()
		plt.clf()

