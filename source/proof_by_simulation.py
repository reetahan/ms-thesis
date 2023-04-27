from graphs import *
from conversions import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import factorial, floor, ceil, cos, sin, acos, sqrt
from itertools import permutations, combinations

def main():

	r = 0.6
	p = 0.6
	a_u = 0.6
	N = 12
	sims = 10000

	results = []
	for i in range(sims):
		sample = test_graph_generation(N,r,p,a_u)
		results.append(sample)
	empirical_res = np.mean(results)
	print('Empirical Result: ' + str(empirical_res))
	theoretical_res = theory_test(N,r,p,a_u)
	print('Theoretical Result: ' + str(theoretical_res))
	
def test_graph_generation(n,r,p,a_u):

	test = BPA2PGraph(n,r,p)
	success = 0
	distribution = []
	for node in test.graph.nodes:
		neighbors = test.graph.neighbors(node)
		denominator = test.graph.degree[node]
		numerator = 0
		for neighbor in neighbors:
			if(test.color_map[neighbor] == 'red'):
				numerator += 1
		red_frac= numerator/denominator
		if(red_frac >=  a_u):
			success += 1
	result = success/n
	return result

def theory_test(n,r,p,a_u):
	
	perms, red_cts = generate_all_permutations(n-2,['R','B'])
	perm_len = len(perms)
	return np.sum([q/n * prob_q(n,r,p,a_u,q,perms,red_cts,perm_len) for q in range(1,n+1)])

def prob_q(n,r,p,a_u,q,perms,red_cts,perm_len):
	across_all_perms = 0
	for perm_idx in range(len(perms)):
		perm = [x for x in perms[perm_idx]]
		for i in range(len(perm)):
			perm[i] = perm[i] + str(i+1)
		red_ct = red_cts[perm_idx]
		q_assignments = generate_all_q_assignments(q,n,perm)
		across_all_q_asgs = 0
		for q_asg_idx in range(len(q_assignments)):
			'''
				We now have a particular Q-assignment ex:(R0:Q,B0:Q,R1:NQ,B2:Q,B3:NQ,R4:NQ,B5:Q) for our given permutation ex:(R0,B0,R1,B2,B3,R4,B5).

				We use generate_event_series to generate a set of all possible event series 
				ex: {[(R0:B0,B0:R0,R1:R0,B2:R1,B3:B0,R4:R0,B5:B0),(R0:B0,B0:R0,R1:R0,B2:R0,B3:B2,R4:R1,B5:B2),]...}
				that satisfy our particular Q-assignment.

				Then we get the probability of each event series in our set occuring to form G, and sum them, to get the overall probability of our Q-assignment.
				We add this to our across_all_q_asgs counter that is added to as we loop across all possible Q-assignments for our given permutation.

				Then, once this counter contains the sum of the possiblity of all Q-assignments for our permutation, we weight this value by the probability of
				the permutation occuring at all, and it add it to our across_all_perms counter.

				This across_all_perms eventually sums across all permutations of possibilities for G, and returns it as the overall possiblity of there being
				q nodes out of the n being Q-nodes following graph creation (guaranteed to turn red after 1 iteration). The probability of a random node being a Q-node
				is computed by summing over the probability of all possible q values times their q/n.

			'''
			possible_events = generate_event_series(q_assignments[q_asg_idx],a_u)
			across_all_q_asgs += np.sum([events_probability(possible_event, p, r) for possible_event in possible_events])

		across_all_perms += across_all_q_asgs
	print(f'For Q = {q}, we have likelihood of {across_all_perms}')
	return across_all_perms

def generate_event_series(q_asg, a_u):
	nodes = list(q_asg.keys())
	assignment_template = [['R0','B0'],['B0','R0']]
	rest_to_assign_nodes = nodes[2:]
	for node in rest_to_assign_nodes:
		unassigned = 'UR' if node[0] == 'R' else 'UB'
		assignment_template.append([node,unassigned])

	cur_possible_events = [assignment_template]

	'''
	on iter accept all possible lists. for each list, check if you have been assigned.
	if not, generate possibilities by assigning 1 earlier node. then try, assigning later nodes,
	by iterating  through 0 - remaining possible new neighbors, and appropriate alpha number of red node assignments.
	return these
	'''
	for index in range(len(nodes)):
		cur_possible_events = generate_event_step(cur_possible_events, q_asg, a_u, index, nodes)
		if(len(cur_possible_events) == 0):
			break

	return cur_possible_events

def generate_event_step(cur_possible_events, q_asg, a_u, index, nodes):
	N = len(nodes)

	new_events = []
	for cur_event in cur_possible_events:
		#We only keep scenarios that have set the current node's joining node upon arrival of processing
		if(cur_event[index][1][0] == 'U'):
			continue

		q_value = q_asg[cur_event[index][0]]
		options = []
		additional = []

		red_choices = [nodes[index+1:][i] for i in range(len(nodes[index+1:])) if (nodes[index+1:][i][0] == 'R' and cur_event[index+1+i][1] in ['UR','UB'] )]
		total_red_left = len(red_choices)
		blue_choices = [nodes[index+1:][i] for i in range(len(nodes[index+1:])) if (nodes[index+1:][i][0] == 'B' and cur_event[index+1+i][1] in ['UR','UB'])]
		total_blue_left = len(blue_choices)

		if(cur_event[index][1][0] == 'R'):
			if(q_value == 'Q'):

				for poss_red_add in range(0,total_red_left + 1):
					max_poss_blue_add = int((poss_red_add + 1)/a_u) - (poss_red_add + 1)
					for poss_blue_add in range(0, min(max_poss_blue_add + 1, total_blue_left + 1)):
						additional.append((poss_red_add,poss_blue_add))
				for pair in additional:
					option = {'event': cur_event, 'cur_node': cur_event[index][0], 'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_choices}
					options.append(option)
				
			else:
				for poss_blue_add in range(0,total_blue_left + 1):
					max_poss_red_add = int(poss_blue_add/(1 - a_u)) - poss_blue_add - 1
					for poss_red_add in range(0, min(max_poss_red_add + 1, total_red_left + 1)):
						additional.append((poss_red_add,poss_blue_add))
				for pair in additional:
					option = {'event': cur_event, 'cur_node': cur_event[index][0], 'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_choices}
					options.append(option)
		else:
			if(q_value == 'Q'):
				for poss_red_add in range(0,total_red_left + 1):
					max_poss_blue_add = int(poss_red_add/a_u) - poss_red_add - 1
					for poss_blue_add in range(0, min(max_poss_blue_add + 1, total_blue_left + 1)):
						additional.append((poss_red_add,poss_blue_add))
				for pair in additional:
					option = {'event': cur_event, 'cur_node': cur_event[index][0],'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_choices}
					options.append(option)
				
			else:
				for poss_blue_add in range(0,total_blue_left + 1):
					max_poss_red_add = int((poss_blue_add + 1)/(1 - a_u)) - (poss_blue_add + 1)
					for poss_red_add in range(0, min(max_poss_red_add + 1, total_red_left + 1)):
						additional.append((poss_red_add,poss_blue_add))
				for pair in additional:
					option = {'event': cur_event, 'cur_node': cur_event[index][0], 'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_choices}
					options.append(option)
				

		for option in options:
			new_events += generate_event_permutations(option)

	return new_events

def generate_event_permutations(option):
	possiblities = []
	red_combos = list(combinations(option['red_options'], option['num_red']))
	blue_combos = list(combinations(option['blue_options'], option['num_blue']))
	total_combos = []
	for red_combo in red_combos:
		for blue_combo in blue_combos:
			total_combos.append(red_combo + blue_combo)

	for combo in total_combos:
		new_possibility = [list(x) for x in option['event']]
		for updated in combo:
			new_possibility[int(updated[1])+1][1] = option['cur_node']
		possiblities.append(new_possibility)

	return possiblities

			

def events_probability(event,p,r):

	prob = 1
	degs = {'R0': 1,'B0': 1}
	red_degsum = 1
	for i in range(2,len(event)):
		combo = (event[i][0][0],event[i][1][0])

		degsum = 2*(i-1)
		deg_frac = degs[event[i][1]]/degsum
		red_degfrac = red_degsum/degsum
		blue_degfrac = 1 - red_degfrac

		if(combo == ('B','R')):
			cur_prob = deg_frac*(1-r)*p/(1 - red_degfrac*(1-p))
		if(combo == ('B','B')):
			cur_prob = deg_frac*(1-r)/(1 - red_degfrac*(1-p))
		if(combo == ('R','R')):
			cur_prob = deg_frac*r/(1 - blue_degfrac*(1-p))
		if(combo == ('R','B')):
			cur_prob = deg_frac*p*r/(1 - blue_degfrac*(1-p))

		degs[event[i][0]] = 1
		degs[event[i][1]] += 1
		if(combo[0] == 'R'):
			red_degsum += 1
		if(combo[1] == 'R'):
			red_degsum += 1

		prob *= cur_prob
	
	return prob


def generate_all_q_assignments(q,n,perm):
	base = ['R0','B0']
	l1_base = ('NQ','NQ')
	l2_base = ('Q','Q')
	l3_base = ('Q','NQ')
	l4_base = ('NQ','Q')
	l1_ =  ['Q' for i in range(q)] + ['NQ' for i in range(n-2-q)]
	final_l1 = [list(l1_base + x) for x in set(permutations(l1_))]
	l2_ =  ['Q' for i in range(q-2)] + ['NQ' for i in range(n-2-q+2)]
	final_l2 = [list(l2_base + x) for x in set(permutations(l2_))]
	l3_ =  ['Q' for i in range(q-1)] + ['NQ' for i in range(n-2-q+1)]
	final_l3 = [list(l3_base + x) for x in set(permutations(l3_))]
	l4_ =  ['Q' for i in range(q-1)] + ['NQ' for i in range(n-2-q+1)]
	final_l4 = [list(l4_base + x) for x in set(permutations(l4_))]
	final = final_l1 + final_l2 + final_l3 + final_l4
	final = [cand_final for cand_final in final if cand_final.count('Q') == q and len(cand_final) == n]
	
	perm = base + perm
	poss_assignments = [dict(zip(perm,q_asg)) for q_asg in final]

	return poss_assignments


def generate_all_permutations(N,colors):
	permutes = []
	red_cts = []
	for red_ct in range(0,N+1):
		cur_ls = ['R' for i in range(red_ct)] + ['B' for i in range(N-red_ct)]
		possible_permutes = [list(x) for x in set(permutations(cur_ls))]
		permutes += list(possible_permutes)
		red_cts += [red_ct] * len(possible_permutes)
	return permutes, red_cts


if __name__ == '__main__':
	main()