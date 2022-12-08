import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt

def sb_model_two(n,p_r, p_b,rho_col, rho_res):
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
            

def true_bpa_two(n, p_r,p_b, rho_col):
    p_b = 1 - p_r 

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

def color(node, red, blue):
    if(node in red):
        return "red"
    if(node in blue):
        return "blue"

def graph_analysis(G, red, blue):
    red_frac = {"red":[],"blue":[]}
    blue_frac = {"red":[],"blue":[]}
    color_conn = {"red": red_frac,"blue":blue_frac}
    for n in G.nodes:
        my_color = color(n, red, blue, green)
        neighbor_colors = np.array([color(n,red,blue) for n in G.neighbors(n)])
        red_f = float(np.count_nonzero(neighbor_colors == "red"))/len(neighbor_colors)
        blue_f = float(np.count_nonzero(neighbor_colors == "blue"))/len(neighbor_colors)
        color_conn[my_color]["red"].append(red_f)
        color_conn[my_color]["blue"].append(blue_f)

    for c in red_frac:
        red_frac[c] = np.mean(red_frac[c])
    for c in blue_frac:
        blue_frac[c] = np.mean(blue_frac[c])
    print("Red")
    print(red_frac)
    print("Blue")
    print(blue_frac)
    return

def conversions(G,red,blue,conv_thresh, color_map):
    '''
    r_b = 0
    b_r = 0
    degs_rb = []
    degs_br = []
    '''
    for n in G.nodes:

        neighbors = G.neighbors(n)
        my_color = color(n,red,blue)
        cur_color = my_color

        if(my_color == 'red'):
            my_color = red
        else:
            my_color = blue
        neighbor_colors = np.array([color(v,red,blue) for v in neighbors])
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

        '''
        if(selection == 'red' and my_color == blue):
            r_b += 1
            degs_br.append(len(neighbor_colors))

        if(selection == 'blue' and my_color == red):
            b_r += 1
            degs_rb.append(len(neighbor_colors))
        '''

        if(selection == 'red'):
            my_color.remove(n)
            color_map[n] = 'red'
            red.add(n)

        if(selection == 'blue'):
            my_color.remove(n)
            color_map[n] = 'blue'
            blue.add(n)

    '''
    degs_rb_d = {}
    degs_br_d = {}
    for x in degs_rb:
        if(x not in degs_rb_d):
            degs_rb_d[x] = 1
        else:
            degs_rb_d[x] += 1
    for x in degs_br:
        if(x not in degs_br_d):
            degs_br_d[x] = 1
        else:
            degs_br_d[x] += 1
    #degs_rb_d = sorted(degs_rb_d)
    #degs_br_d = sorted(degs_br_d)
    '''

    return red,blue,color_map #, r_b, b_r, degs_rb_d, degs_br_d

def rd3(num):
    return round(num,3)

def main():

    '''
    info = {"conv_thresh":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Conversion Threshold (5% to 95%)"}, "rho_col":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Rho col (5% to 95%)"}, 
    "init_ratio":{"vals":np.arange(0.5,0.60,0.005)[::-1], "xlabel": "Initial Red-Blue Difference (50-50 to 40-60)"}}
    other_info = {"conv_thresh": [0.40,0.65], "rho_col": [0.25,0.75], "init_ratio":[(0.5,0.5),(0.65,0.35)]}
    '''

    '''
    info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"}, 
    "init_ratio":{"vals":np.arange(0.5,0.60,0.005)[::-1], "xlabel": "Initial Red-Blue Difference (50-50 to 40-60)"}}
    other_info = {"conv_thresh": [0.35, 0.45,0.55,0.75], "rho_col": [0.25, 0.45, 0.55, 0.75], "init_ratio":[(0.55,0.45), (0.8,0.2)]}
    #other_info = {"conv_thresh": [0.34, 0.45,0.55,0.75], "rho_col": [0.55], "init_ratio":[(0.55,0.45), (0.8,0.2)]}
    '''

    info = {"conv_thresh":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Conversion Threshold (0% to 100%)"}}#, "rho_col":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho col (0% to 100%)"}, 
    #"rho_res":{"vals":np.arange(0.00,1.01,0.05), "xlabel": "Rho res (0% to 100%)"}}
    other_info = {"conv_thresh": [0.55], "rho_col": [0.6],"rho_res": [0.4], "init_ratio":[(0.55,0.45),(0.8,0.2)]}


    y_lab = 'Final Proportions of Parties'
    n = 5000
    iter_max = 5

    for question in info:
        print(question)
        #if(question != 'conv_thresh'): #and question != 'rho_col'):
        #    continue
        list_of_others = []
        cores_map = []
        x_lab = info[question]["xlabel"]
        question_range = info[question]["vals"]

        for other in other_info:
            if(other != question):
                list_of_others.append(other_info[other])
                cores_map.append(other)
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
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", rho_res: " + str(rd3(rho_res))   + ", conv_thresh: " + str(rd3(conv_thresh)) 
            if(rho_res_idx == None):
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", rho_col: " + str(rd3(rho_col)) + ", conv_thresh: " + str(rd3(conv_thresh)) 
            if(p_r_idx == None):
                title_str = "N: " + str(n)  + ", conv_thresh: " + str(rd3(conv_thresh)) + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res)) 
            if(conv_thresh_idx == None):
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res)) 
            
            init_diffs = []
            red_p = []
            blue_p = []
    
            for var_choice in question_range:
                if(rho_col_idx == None):
                    rho_col = var_choice
                    init_str = title_str + ", rho_col:" + str(rd3(rho_col)) 
                    init_diffs.append(100*(rho_col))
                    fname = "SB_2p_mass_combos_RHOCOL_" + "_prb_" + "rr_" + str(rd3(rho_res)) + str((rd3(p_r),rd3(p_b))) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(rho_res_idx == None):
                    rho_res = var_choice
                    init_str = title_str + ", rho_res:" + str(rd3(rho_res)) 
                    init_diffs.append(100*(rho_res))
                    fname = "SB_2p_mass_combos_RHORES_" + "_prb_" + str((rd3(p_r),rd3(p_b))) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(p_r_idx == None):
                    p_r = var_choice
                    p_b = 1 - p_r
                    init_str = title_str + ", p_r:" + str(rd3(p_r)) + ", p_b:" + str(rd3(p_b)) 
                    init_diffs.append(100*(np.abs(p_r-p_b)))
                    fname = "SB_2p_mass_combos_INITRAT_" + "rc_" + str(rd3(rho_col)) + "rr_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(conv_thresh_idx == None):
                    conv_thresh = var_choice
                    init_str = title_str + ", conv_thresh:" + str(rd3(conv_thresh))
                    init_diffs.append(100*(conv_thresh)) 
                    fname = "SB_2p_mass_combos_CONVTHRESH_" + "rc_" + str(rd3(rho_col)) + "rr_" + str(rd3(rho_res)) + "_prb_" + str((rd3(p_r),rd3(p_b))) + ".png"

                print(init_str, flush=True)

                red_avg = []
                blue_avg = []

                for seed in np.arange(0,32,3):

                    np.random.seed(seed)
                        
                    cur_reds = []
                    cur_blues = []


                    G, red, blue, color_map = sb_model_two(n, p_r, p_b, rho_col,rho_res)

                    init_red = rd3(len(red)/len(G.nodes)*100)
                    init_blue = rd3(len(blue)/len(G.nodes)*100)

                    cur_reds.append(init_red)
                    cur_blues.append(init_blue)

                    
                    iters = 0
                    while(iters < iter_max):
                        red,blue,color_map = conversions(G,red,blue,conv_thresh,color_map)

                        '''
                        red,blue,color_map, rb,br,deg_rb,deg_br = conversions(G,red,blue,conv_thresh,color_map)
                        
                        print('===========================')
                        print('seed: ' + str(seed) + ', iter: ' + str(iters))
                        print('R->B: ' + str(rb))
                        print('B->R: ' + str(br))
                        print('Degs r->b: ' + str(deg_rb))
                        print('Degs b->r: ' + str(deg_br))
                        print('===========================')
                        '''
                        iters = iters + 1
                        cur_reds.append(rd3(len(red)/len(G.nodes)*100))
                        cur_blues.append(rd3(len(blue)/len(G.nodes)*100))
                    
                    print(seed, flush=True)
                    
                    red_avg.append(cur_reds[len(cur_reds)-1])
                    blue_avg.append(cur_blues[len(cur_blues)-1])

                print('Red for conv=' + str(conv_thresh) + ' : ' + str(rd3(np.average(red_avg))))
                print('Blue for conv=' + str(conv_thresh) + ' : ' + str(rd3(np.average(blue_avg))))
                red_p.append(rd3(np.average(red_avg)))
                blue_p.append(rd3(np.average(blue_avg)))

            plt.rcParams.update({'font.size': 11})
            plt.plot(init_diffs, red_p,color='red')
            plt.plot(init_diffs, blue_p,color='blue')
            plt.xlabel(x_lab)
            plt.ylabel(y_lab)
            plt.title(title_str)
            plt.savefig(fname)
            plt.clf()
        

    

if __name__ == '__main__':
    main()