from graphs import *
from conversions import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import factorial, floor, ceil, cos, sin, acos, sqrt
from itertools import permutations

def main():

	r = 0.8
	p = 0.8
	a_u = 0.4
	N = 10
	sims = 1000

	empirical_res = np.mean([test_graph_generation(N,r,p,a_u) for i in range(sims)])
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
	print('Generated permutations')

	return np.sum([q/n * prob_q(n,r,p,a_u,q) for q in range(1,n+1)])

def prob_q(n,r,p,a_u,q,perms,red_cts,perm_len):
	across_all_perms = 0
	for perm_idx in range(len(perms)):
		perm = perms[perm_idx]
		for i in range(len(perm)):
			perm[i] = perm[i] + str(i+1)
		red_ct = red_cts[perm_idx]
		odds_of_perm =  r**red_ct * (1-r)**(n-2-red_ct)
		q_assignments = generate_all_q_assignments(q,n,perm)
		print('Generated Q assignments for permutation ' + str(perm_idx+1) + ' of ' + str(perm_len) + ' in Q=' + str(q))
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
			across_all_q_asgs += np.sum([events_probability(possible_event, r, p) for possible_event in possible_events])
		across_all_perms += odds_of_perm * across_all_q_asgs
	return across_all_perms

def generate_event_series(q_asg, a_u):
	nodes = list(q_asg.keys())
	assignment_template = [('R0','B0'),('B0','R0')]
	rest_to_assign_nodes = nodes[2:]
	for node in rest_to_assign_nodes:
		unassigned = 'UR' if node[0] == 'R' else 'UB'
		assignment_template.append((node,unassigned))

	cur_possible_events = [assignment_template]

	'''
	on iter accept all possible lists. for each list, check if you have been assigned.
	if not, generate possibilities by assigning 1 earlier node. then try, assigning later nodes,
	by iterating  through 0 - remaining possible new neighbors, and appropriate alpha number of red node assignments.
	return these
	'''
	for i in range(len(nodes)):
		cur_possible_events = generate_event_step(cur_possible_events, q_asg, a_u, 
			index, nodes)
		if(len(cur_possible_events) == 0):
			break

	return cur_possible_events

def generate_event_step(cur_possible_events, q_asg, a_u, index, nodes):
	q_value = q_asg[cur_event[index][0]]
	N = len(nodes)

	new_events = []
	for cur_event in cur_possible_events:
		options = []
		additional = []

		red_choices = [nodes[index+1:][i] for i in range(len(nodes[index+1:])) if (nodes[index+1:][i] == 'R' and cur_event[index+1+i][1] in ['UR','UB'] )]
		total_red_left = len(red_choices)
		blue_choices = [nodes[index+1:][i] for i in range(len(nodes[index+1:])) if (nodes[index+1:][i] == 'B' and cur_event[index+1+i][1] in ['UR','UB'])]
		total_blue_left = len(blue_choices)

		init_color = None
		past_color_options = None
		if(cur_event[index][1] in ['UR','UB']):
			init_color = ['R','B']
			past_color_options = [[node for node in nodes[:index] if node[0] == 'R'], [node for node in nodes[:index] if node[0] == 'B']]

		if(init_color is None):
			if(cur_event[index][1][0] == 'R'):
				if(q_value == 'Q'):
					min_add = 
					max_add =

					for pair in additional:
						option = {'event': cur_event, 'init_color': None, 'past_color_options': None, 'num_red': pair[0], \
						'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
						options.append(option)
				else:
					min_add = 
					max_add =

					for pair in additional:
						option = {'event': cur_event, 'init_color': None, 'past_color_options': None, 'num_red': pair[0], \
						'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
						options.append(option)
			else:
				if(q_value == 'Q'):
					min_add = 
					max_add =

					for pair in additional:
						option = {'event': cur_event, 'init_color': None, 'past_color_options': None, 'num_red': pair[0], \
						'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
						options.append(option)
				else:
					min_add = 
					max_add =

					for pair in additional:
						option = {'event': cur_event, 'init_color': None, 'past_color_options': None, 'num_red': pair[0], \
						'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
						options.append(option)

		else:
			if(q_value == 'Q'):
				min_add = 
				max_add =

				for pair in additional:
					option = {'event': cur_event, 'init_color': init_color[0], 'past_color_options': past_color_options[0], \
					'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
					options.append(option)

				additional = []

				min_add = 
				max_add =

				for pair in additional:
					option = {'event': cur_event, 'init_color': init_color[1], 'past_color_options': past_color_options[1], \
					'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
					options.append(option)
			else:
				min_add = 
				max_add =

				for pair in additional:
					option = {'event': cur_event, 'init_color': init_color[0], 'past_color_options': past_color_options[0], \
					'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
					options.append(option)

				additional = []

				min_add = 
				max_add =

				for pair in additional:
					option = {'event': cur_event, 'init_color': init_color[1], 'past_color_options': past_color_options[1], \
					'num_red': pair[0], 'red_options':red_choices, 'num_blue': pair[1], 'blue_options': blue_options}
					options.append(option)

		
		for option in options:
			new_events += generate_event_permutations(option)

	return new_events

def generate_event_permutations(option):
	possiblities = []

	return possiblities

			

def events_probability(event,r,p):
	pass

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































def test_ccdf_generator(n, r, p, a_u):
	result = 0
	b_est_roots = cubic_solve(4*p-2*p**2-2,2 + 3*p**2 - 5*p + 2*r - 2*p*r,2*p - 2*r + 2*p*r - p**2,-1*r*p)
	#b_est = [x for x in b_est_roots if x >= 0 and x <= 1][0]
	b_est = 0.2
	for d in range(0,n):
		print('d: ' + str(d) +' of ' + str(n))

		inner_result_one_one = 0
		inner_result_one_two = 0

		for i in range(1,n):

			inner_result_two_one = 0
			inner_result_two_two = 0


			for j in range(1, min(floor((d-2)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceil(j*a_u)-1,j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)


					for m in range(1, int(factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k)))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]

						for w in range(1, n-i):

							psmw = psmw_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_one = inner_result_two_one + inner_result_three

			for j in range(1, min(floor((d-1)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceil(j*a_u),j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, int(factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k)))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]

						for w in range(1, n-i):

							psmw = psmw_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_two = inner_result_two_two + inner_result_three

			coefficient_one = b_est/(1 - ((1-b_est)*(1-p)))
			coefficient_two = ((1-b_est)*p)/(1 - ((1-b_est)*(1-p)))
			inner_result_one_one = inner_result_one_one + coefficient_one*inner_result_two_one + coefficient_two*inner_result_two_two

		for i in range(1,n):

			
			inner_result_two_one = 0
			inner_result_two_two = 0


			for j in range(1, min(floor((d-1)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceil(j*a_u)-1,j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, int(factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k)))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]
		
						for w in range(1, n-i):

							psmw = psmw_prime_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_one = inner_result_two_one + inner_result_three

			for j in range(1, min(floor((d)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceil(j*a_u),j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, int(factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k)))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]
						
						for w in range(1, n-i):

							psmw = psmw_prime_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_two = inner_result_two_two + inner_result_three

			coefficient_one = b_est/(1 - ((b_est)*(1-p)))
			coefficient_two = ((1-b_est)*p)/(1 - ((b_est)*(1-p)))
			inner_result_one_two = inner_result_one_two + coefficient_one*inner_result_two_one + coefficient_two*inner_result_two_two



		coefficient = r**d * (1-r)**(n-d)/(factorial(d) * factorial(n-d))
		result = result + coefficient*(d*inner_result_one_one + (n-d)*inner_result_one_two)

	result = factorial(n-1) * result
	return result

def permutation_generator(N, B, R):
	permute_str = ""
	for i in range(N):
		permute_str += "N"
	for i in range(B):
		permute_str += "B"
	for i in range(R):
		permute_str += "R"
	permutation_list = [''.join(x) for x in permutations(permute_str)]
	return permutation_list

def permutation_counters(permutations):
	pcs = []
	for permutation in permutations:
		cur_pc = ''
		counter = 1
		for letter in permutation:
			cur_pc += str(counter)
			if(letter != 'N'):
				counter += 1
		pcs.append(cur_pc)
	return pcs

def psmw_generator(p,r,w,i,b_est,cur_permutation,cur_pc):
	cur_event = cur_permutation[w]
	h_w = int(cur_pc[w])

	if(cur_event == 'R'):
		psnw = (r*h_w)/((2*(w+i))*(1-((1-b_est)*(1-p))))

	if(cur_event == 'B'):
		psnw = ((1-r)*p*h_w)/((2*(w+i))*(1-((b_est)*(1-p))))

	if(cur_event == 'N'):
		psnw_r = (r*h_w)/((2*(w+i))*(1-((1-b_est)*(1-p))))
		psnw_b = ((1-r)*p*h_w)/((2*(w+i))*(1-((b_est)*(1-p))))
		psnw = 1 - psnw_r - psnw_b

	return psnw

def psmw_prime_generator(p,r,w,i,b_est,cur_permutation,cur_pc):
	cur_event = cur_permutation[w]
	h_w = int(cur_pc[w])

	if(cur_event == 'R'):
		psnw = (p*r*h_w)/((2*(w+i))*(1-((1-b_est)*(1-p))))

	if(cur_event == 'B'):
		psnw = ((1-r)*h_w)/((2*(w+i))*(1-((b_est)*(1-p))))

	if(cur_event == 'N'):
		psnw_r = (p*r*h_w)/((2*(w+i))*(1-((1-b_est)*(1-p))))
		psnw_b = ((1-r)*h_w)/((2*(w+i))*(1-((b_est)*(1-p))))
		psnw = 1 - psnw_r - psnw_b

	return psnw


def cubic_solve(a, b, c, d):

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

    elif (a == 0):                              # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0:
            D = sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)
            
        return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])              # Returning Equal Roots as numpy array.

    elif h <= 0:                                # All 3 roots are Real

        i = sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = cos(k / 3.0)                   # Helper Temporary Variable
        N = sqrt(3) * sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * sqrt(3) * 0.5j

        return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

if __name__ == '__main__':
	main()