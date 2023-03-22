from graphs import *
from conversions import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from math import factorial, floor, ceil
from itertools import permutations

def main():

	n = 10
	r = 0.6
	p = 0.3
	a_u = 0.5

	empirical = test_graph_generation(n,r,p,a_u)
	theoretical =  test_ccdf_generator(n,r,p,a_u)

	print('Theoretical Probability: ' + str(theoretical))
	print('Empirical Probability: ' + str(empirical))

def test_graph_generation(n,r,p,a_u):

	test = BPA2PGraph(n,r,p)
	success = 0
	distribution = []
	for node in test.graph.nodes:
		neighbors = test.graph.neighbors(node)
		denominator = len(neighbors)
		numerator = 0
		for neighbor in neighbors:
			if(test.color_map[neighbor] == 'red'):
				numerator += 1
		red_frac= numerator/denominator
		if(red_frac >=  a_u):
			success += 1
	result = success/n
	return result


def test_ccdf_generator(n, r, p, a_u):
	result = 0
	b_est = cubic_solve(4*p-2*p**2-2,2 + 3*p**2 - 5*p + 2*r - 2*p*r,2*p - 2*r + 2*p*r - p**2,-1*r*p)

	for d in range(0,n):

		inner_result_one_one = 0
		inner_result_one_two = 0

		for i in range(1,n):

			inner_result_two_one = 0
			inner_result_two_two = 0


			for j in range(0, min(floor((d-2)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceiling(j*a_u)-1,j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]

						for w in range(1, n-i):

							psmw = psmw_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_one = inner_result_two_one + inner_result_three

			for j in range(0, min(floor((d-1)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceiling(j*a_u),j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]

						for w in range(1, n-i):

							psmw = psmw_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_two = inner_result_two_two + inner_result_three

			coefficient_one = b_est/(1 - ((1-b_est)(1-p)))
			coefficient_two = ((1-b_est)*p)/(1 - ((1-b_est)(1-p)))
			inner_result_one_one = inner_result_one_one + coefficient_one*inner_result_two_one + coefficient_two*inner_result_two_two

		for i in range(1,n):

			
			inner_result_two_one = 0
			inner_result_two_two = 0


			for j in range(0, min(floor((d-1)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceiling(j*a_u)-1,j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]
		
						for w in range(1, n-i):

							psmw = psmw_prime_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_one = inner_result_two_one + inner_result_three

			for j in range(0, min(floor((d)/a_u),n-i)):
				inner_result_three = 0

				for k in range(ceiling(j*a_u),j):
					inner_result_four = 0
					permutations = permutation_generator(n-i-j,j-k,k)
					perm_counts = permutation_counters(permutations)

					for m in range(1, factorial(n-i)/(factorial(n-i-j)*factorial(j-k)*factorial(k))):
						inner_result_five = 1
						cur_permutation = permutations[m]
						cur_pc = perm_counts[m]
						
						for w in range(1, n-i):

							psmw = psmw_prime_generator(p,r,w,i,b_est,cur_permutation,cur_pc)

							inner_result_five = inner_result_five * psmw

						inner_result_four = inner_result_four + inner_result_five

					inner_result_three = inner_result_three + inner_result_four

				inner_result_two_two = inner_result_two_two + inner_result_three

			coefficient_one = b_est/(1 - ((b_est)(1-p)))
			coefficient_two = ((1-b_est)*p)/(1 - ((b_est)(1-p)))
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
	permutations = [''.join(x) for x in permutations(permute_str)]
	return permutations

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
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
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

        i = math.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = math.cos(k / 3.0)                   # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

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