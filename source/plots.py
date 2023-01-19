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

		print(self.file_list)
		print(self.title_list)
		print(self.xlabel)
		print(self.ylabel)

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

'''
def main():
	#file_list = ['3p_BPA_results/' + x for x in os.listdir('3p_BPA_results')]
	file_list = ['2p_BPA_results/' + x for x in os.listdir('2p_BPA_results')]
	#file_list = ['2p_SB_results/' + x for x in os.listdir('2p_SB_results')]
	#file_list = ['serial_2p_BPA_results/' + x for x in os.listdir('serial_2p_BPA_results')]
	#file_list = ['serial_2p_SB_results/' + x for x in os.listdir('serial_2p_SB_results')]

	for file_ in file_list:
		fparts = file_.split('_')
		xlabel = 'Conversion Threshold'
		if(fparts[5] == 'RHORES'):
			xlabel = 'Rho res'
		if(fparts[5] == 'RHOCOL'):
			xlabel = 'Rho col'
		ratio = fparts[8]
		rhocol = None 
		rhores = None 
		conv_thresh = None 
		if(fparts[6] == 'rc'):
			rhocol = fparts[7]
			ratio = fparts[9].split('.txt')[0]
		if(fparts[9] == 'rhocol'):
			rhocol = fparts[10]
		if(fparts[9] == 'rhores'):
			rhores = fparts[10]
		if(fparts[9] == 'convthresh'):
			conv_thresh = fparts[10].split('.txt')[0]
		if(len(fparts) > 11 and fparts[11] == 'rhocol'):
			rhocol = fparts[12].split('.')[0]
		if(len(fparts) > 11 and fparts[11] == 'rhores'):
			rhores = fparts[12].split('.')[0]
		if(len(fparts) > 11 and fparts[11] == 'convthresh'):
			conv_thresh = fparts[12].split('.')[0]
		
		#title = "3-party BPA | " + str(xlabel) + " | " + ratio + " | "
		title = "2-party BPA | " + str(xlabel) + " | " + ratio + " | "
		#title = "2-party SB | " + str(xlabel) + " | " + ratio + " | "
		#title = "Serial 2-party BPA | " + str(xlabel) + " | " + ratio + " | "
		#title = "Serial 2-party SB | " + str(xlabel) + " | " + ratio + " | "
		if(rhocol != None):
			title += ' rho_col: ' + str(rhocol)
		if(rhores != None):
			title += ' rho_res: ' + str(rhores)
		if(conv_thresh != None):
			title += ' conv_thresh: ' + str(conv_thresh)

		try:
			f = open(file_,"r")
			lines = f.readlines()
			f.close()
		except:
			continue

		x = np.arange(0,101,5)
		r = [float(lines[i]) for i in range(24,45)]
		b = [float(lines[i]) for i in range(46,67)]
		#g = [float(lines[i]) for i in range(68,89)]

		plt.plot(x,r,color='red',label='Red Party')
		plt.plot(x,b,color='blue',label='Blue Party')
		#plt.plot(x,g,color='green', label = 'Green Party')
		plt.xlabel(xlabel)
		plt.title(title)
		plt.legend()
		new_fig_name = file_.split('.txt')[0] + '.png'
		plt.savefig(new_fig_name)
		plt.clf()
		#plt.show()
'''



if __name__ == '__main__':
	main()