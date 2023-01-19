import matplotlib.pyplot as plt
import numpy as np
import os

class Plot:
	def __init__(self, fnames, titles, party_count, var_range, xlabel, ylabel):
		self.file_list = fnames
		self.party_count = party_count
		self.title_list = titles
		self.var_range = 100*var_range
		self.xlabel = xlabel
		self.ylabel = ylabel

	def generate_plot(self,show):
		for i in range(len(self.file_list)):

			try:
				f = open(self.file_list[i],"r")
				lines = f.readlines()
				f.close()
			except:
				print('Could not open file: ' + self.file_list[i])
				continue

			r = [float(lines[i]) for i in range(3+len(self.var_range),3+2*len(self.var_range))]
			b = [float(lines[i]) for i in range(4+2*len(self.var_range),4+3*len(self.var_range))]
			plt.plot(self.var_range,r,color='red',label='Red Party')
			plt.plot(self.var_range,b,color='blue',label='Blue Party')

			if(self.party_count == 3):
				g = [float(lines[i]) for i in range(5+3*len(self.var_range),5+4*len(self.var_range))]
				plt.plot(x,g,color='green', label = 'Green Party')

			
			plt.xlabel(self.xlabel)
			plt.ylabel(self.ylabel)
			plt.title(self.title_list[i])
			plt.legend()
			new_fig_name = self.file_list[i].split('.txt')[0] + '.png'
			plt.savefig(new_fig_name)
			if(show):
				plt.show()
				plt.clf()