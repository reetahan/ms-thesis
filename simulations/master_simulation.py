import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt


def sb_model_three(n,p_r, p_b,rho_col, rho_res):
    num_red = n * p_r
    num_blue = n - num_red

    G = nx.MultiGraph()

    for i in range(n):
        for j in range(n):
            if(i == j):
                continue
            if((i < num_red and j < num_red) or (i >= num_red and j >= num_red)):
                edge_ind = np.random.binomial(1,rho_col)
            if((i < num_red and j >= num_red) or (i >= num_red and j < num_red)):
                edge_ind = np.random.binomial(1,rho_res)
            if(edge_ind):
                G.add_edge(i,j)
    return G
    
    


def true_bpa_three(n, p_r,p_b, rho_col, rho_res):
    p_g = 1 - p_r - p_b

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

def color(node, red, blue, green):
    if(node in red):
        return "red"
    if(node in blue):
        return "blue"
    else:
        return "green"

def graph_inspector(G,red,blue,green,color_map):
    rb = 0
    br = 0
    rg = 0
    gr = 0
    bg = 0
    gb = 0
    rr = 0
    bb = 0
    gg = 0
    deg_dist = []
    for n in G.nodes():
        my_color = color(n,red,blue,green)
        neighbor_colors = np.unique(np.array([color(n,red,blue,green) for n in G.neighbors(n)]))
        if(len(neighbor_colors) == 1):
            if(my_color == "red"):
                if(neighbor_colors[0] == "red"):
                    rr += 1
                if(neighbor_colors[0] == "blue"):
                    rb += 1
                    deg_dist.append(G.degree[n])
                if(neighbor_colors[0] == "green"):
                    rg += 1
                    deg_dist.append(G.degree[n])

            if(my_color == "blue"):
                if(neighbor_colors[0] == "red"):
                    br += 1
                    deg_dist.append(G.degree[n])
                if(neighbor_colors[0] == "blue"):
                    bb += 1
                if(neighbor_colors[0] == "green"):
                    bg += 1
                    deg_dist.append(G.degree[n])

            if(my_color == "green"):
                if(neighbor_colors[0] == "red"):
                    gr += 1
                    deg_dist.append(G.degree[n])
                if(neighbor_colors[0] == "blue"):
                    gb += 1
                    deg_dist.append(G.degree[n])
                if(neighbor_colors[0] == "green"):
                    gg += 1

    '''
    print('Red Surrounded by Reds: ' + str(rr))
    print('Blue Surrounded by Blues: ' + str(bb))
    print('Green Surrounded by Greens: ' + str(gg))
    print('~~~~~~~~~~~~~~~~~~')
    print('Red Surrounded by Blues: ' + str(rb))
    print('Red Surrounded by Greens: ' + str(rg))
    print('Percent of All Red to Switch Out at T=100%: ' + str((rb+rg)/len(red)*100))
    print('Percent Increase of New Reds Switching In at T=100%: ' + str((br+gr)/len(red)*100))
    print('Blue Surrounded by Reds: ' + str(br))
    print('Blue Surrounded by Greens: ' + str(bg))
    print('Percent of All Blue to Switch Out at T=100%: ' + str((br+bg)/len(blue)*100))
    print('Percent Increase of New Blues Switching In at T=100%: ' + str((rb+gb)/len(blue)*100))
    print('Green Surrounded by Reds: ' + str(gr))
    print('Green Surrounded by Blues: ' + str(gb))
    print('Percent of All Green to Switch Out at T=100%: ' + str((gr+gb)/len(green)*100))
    print('Percent Increase of New Greens Switching In at T=100%: ' + str((rg+bg)/len(green)*100))
    '''

    return (rb+rg)/len(red)*100, (br+gr)/len(red)*100, (br+bg)/len(blue)*100, (rb+gr)/len(blue)*100, (gr+gb)/len(green)*100,(rg+bg)/len(green)*100, deg_dist
    



def graph_analysis(G, red, blue, green):
    red_frac = {"red":[],"blue":[],"green":[]}
    blue_frac = {"red":[],"blue":[],"green":[]}
    green_frac = {"red":[],"blue":[],"green":[]}
    color_conn = {"red": red_frac,"blue":blue_frac,"green":green_frac}
    for n in G.nodes:
        my_color = color(n, red, blue, green)
        neighbor_colors = np.array([color(n,red,blue,green) for n in G.neighbors(n)])
        red_f = float(np.count_nonzero(neighbor_colors == "red"))/len(neighbor_colors)
        blue_f = float(np.count_nonzero(neighbor_colors == "blue"))/len(neighbor_colors)
        green_f = float(np.count_nonzero(neighbor_colors == "green"))/len(neighbor_colors)
        color_conn[my_color]["red"].append(red_f)
        color_conn[my_color]["blue"].append(blue_f)
        color_conn[my_color]["green"].append(green_f)

    for c in red_frac:
        red_frac[c] = np.mean(red_frac[c])
    for c in blue_frac:
        blue_frac[c] = np.mean(blue_frac[c])
    for c in green_frac:
        green_frac[c] = np.mean(green_frac[c])
    print("Red")
    print(red_frac)
    print("Blue")
    print(blue_frac)
    print("Green")
    print(green_frac)
    return

def conversions(G,red,blue,green,conv_thresh, color_map):

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
        neighbor_colors = np.array([color(v,red,blue,green) for v in neighbors])
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

    return red,blue,green, color_map

def rd3(num):
    return round(num,3)

def main():

    '''
    info = {"conv_thresh":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Conversion Threshold (5% to 95%)"}, "rho_col":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Rho col (5% to 95%)"},
    "rho_res":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Rho res (5% to 95%)"}, "init_ratio":{"vals":np.arange(0.3,0.40,0.005)[::-1], "xlabel": "Initial Red-Green Difference (40.5-20-39.5 to 30-20-50)"}}
    other_info = {"conv_thresh": [0.40,0.65], "rho_col": [0.25,0.75], "rho_res":[0.25,0.75], "init_ratio":[(0.4,0.2,0.4),(0.5,0.3,0.2)]}
    

    info = {"conv_thresh":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Conversion Threshold (5% to 95%)"}, "rho_col":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Rho col (5% to 95%)"},
    "rho_res":{"vals":np.arange(0.05,0.95,0.05), "xlabel": "Rho res (5% to 95%)"}, "init_ratio":{"vals":np.arange(0.3,0.40,0.005)[::-1], "xlabel": "Initial Red-Green Difference (40.5-20-39.5 to 30-20-50)"}}
    other_info = {"conv_thresh": [0.45,0.60,0.75], "rho_col": [0.35, 0.55, 0.75], "rho_res":[0.5], "init_ratio":[(0.34,0.33,0.33), (0.25,0.35,0.40)]}
    '''

    info = {"conv_thresh": {"vals":np.arange(1.01,1.07,0.02), "xlabel": "Conversion Threshold (101% to 107%)"}}
    other_info = {"rho_col": [0.75], "rho_res":[0.25], "init_ratio":[(0.4,0.2,0.4)]}
    
    y_lab = 'Final Proportions of Parties'
    n = 6000
    iter_max = 5

    for question in info:
        print(question)
        if(question != 'conv_thresh' and question != 'rho_col'):
            continue
        list_of_others = []
        cores_map = []
        x_lab = info[question]["xlabel"]
        question_range = info[question]["vals"]
        fname = None

        for other in other_info:
            if(other != question):
                list_of_others.append(other_info[other])
                cores_map.append(other)
        total_tuples = []
        for a in list_of_others[0]:
            for b in list_of_others[1]:
                for c in list_of_others[2]:
                    total_tuples.append((a,b,c))


        rho_res_idx = None 
        rho_col_idx = None 
        p_r_idx = None 
        conv_thresh_idx = None

        for i in range(len(cores_map)):
            if(cores_map[i] == "rho_res"):
                rho_res_idx = i 
            if(cores_map[i] == "rho_col"):
                rho_col_idx = i 
            if(cores_map[i] == "init_ratio"):
                p_r_idx = i 
            if(cores_map[i] == "conv_thresh"):
                conv_thresh_idx = i

        for cur_tup in total_tuples:
            print("Current Tuple: " +  str(cur_tup), flush=True)
            if(rho_res_idx != None):
                rho_res = cur_tup[rho_res_idx]
            if(rho_col_idx != None):
                rho_col = cur_tup[rho_col_idx]
            if(p_r_idx != None):
                p_r = cur_tup[p_r_idx][0]
                p_b = cur_tup[p_r_idx][1]
                p_g = cur_tup[p_r_idx][2]
            if(conv_thresh_idx != None):
                conv_thresh = cur_tup[conv_thresh_idx]


            if(rho_res_idx == None):
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g)) + ", conv_thresh: " + str(rd3(conv_thresh)) + ", rho_col: " + str(rd3(rho_col)) 
            if(rho_col_idx == None):
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g)) + ", conv_thresh: " + str(rd3(conv_thresh)) +  ", rho_res: " + str(rd3(rho_res))
            if(p_r_idx == None):
                title_str = "N: " + str(n)  + ", conv_thresh: " + str(rd3(conv_thresh)) + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res))
            if(conv_thresh_idx == None):
                title_str = "N: " + str(n) + ", p_r: " + str(rd3(p_r)) + ", p_b: " + str(rd3(p_b)) + ", p_g: " + str(rd3(p_g)) + ", rho_col: " + str(rd3(rho_col)) + ", rho_res: " + str(rd3(rho_res))
            
            init_diffs = []
            red_p = []
            blue_p = []
            green_p =[]
            for var_choice in question_range:
                if(rho_res_idx == None):
                    rho_res = var_choice
                    init_str = title_str + ", rho_res:" + str(rd3(rho_res)) 
                    init_diffs.append(100*(rho_res))
                    fname = "mass_combos_RHORES_" + "rc_" + str(rd3(rho_col)) + "_prbg_" + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(rho_col_idx == None):
                    rho_col = var_choice
                    init_str = title_str + ", rho_col:" + str(rd3(rho_col)) 
                    init_diffs.append(100*(rho_col))
                    fname = "mass_combos_RHOCOL_" + "rr_" + str(rd3(rho_res)) + "_prbg_" + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(p_r_idx == None):
                    p_r = var_choice
                    p_b = 0.20
                    p_g = 0.80 - p_r
                    init_str = title_str + ", p_r:" + str(rd3(p_r)) + ", p_b:" + str(rd3(p_b)) + ", p_g:" + str(rd3(p_g)) 
                    init_diffs.append(100*(np.abs(p_g-p_r)))
                    fname = "mass_combos_INITRAT_" + "rc_" + str(rd3(rho_col)) + "_rr_" + str(rd3(rho_res)) + "_convthresh_" + str(rd3(conv_thresh)) + ".png"
                if(conv_thresh_idx == None):
                    conv_thresh = var_choice
                    init_str = title_str + ", conv_thresh:" + str(rd3(conv_thresh))
                    init_diffs.append(100*(conv_thresh)) 
                    fname = "mass_combos_CONVTHRESH_" + "rc_" + str(rd3(rho_col)) + "_prbg_" + str((rd3(p_r),rd3(p_b),rd3(p_g))) + "_rr_" + str(rd3(rho_res)) + ".png"

                print(init_str, flush=True)

                red_avg = []
                blue_avg = []
                green_avg = []

                '''
                rloss_avg = []
                rgain_avg = []
                bloss_avg = []
                bgain_avg = []
                gloss_avg = []
                ggain_avg = []
                

                overall_deg_dist = []
                '''
                for seed in np.arange(0,71,3):

                    np.random.seed(seed)
                        
                    cur_reds = []
                    cur_blues = []
                    cur_greens = []


                    G, red, blue, green, color_map = true_bpa_three(n, p_r, p_b, rho_col, rho_res)

                    
                    '''
                    rloss, rgain, bloss, bgain, gloss, ggain, deg_dist = graph_inspector(G,red,blue,green,color_map)
                    rloss_avg.append(rloss)
                    rgain_avg.append(rgain)
                    bloss_avg.append(bloss)
                    bgain_avg.append(bgain)
                    gloss_avg.append(gloss)
                    ggain_avg.append(ggain)
                    overall_deg_dist += deg_dist
                    '''

                    init_red = rd3(len(red)/len(G.nodes)*100)
                    init_blue = rd3(len(blue)/len(G.nodes)*100)
                    init_green = rd3(len(green)/len(G.nodes)*100)

                    cur_reds.append(init_red)
                    cur_blues.append(init_blue)
                    cur_greens.append(init_green)

                    
                    iters = 0
                    while(iters < iter_max):
                        red,blue,green,color_map = conversions(G,red,blue,green,conv_thresh,color_map)
                        iters = iters + 1
                        cur_reds.append(rd3(len(red)/len(G.nodes)*100))
                        cur_blues.append(rd3(len(blue)/len(G.nodes)*100))
                        cur_greens.append(rd3(len(green)/len(G.nodes)*100))
                    
                    print(seed, flush=True)
                    
                    red_avg.append(cur_reds[len(cur_reds)-1])
                    blue_avg.append(cur_blues[len(cur_blues)-1])
                    green_avg.append(cur_greens[len(cur_greens)-1])

                red_p.append(rd3(np.average(red_avg)))
                blue_p.append(rd3(np.average(blue_avg)))
                green_p.append(rd3(np.average(green_avg)))
                
                '''
                print('Final prop for conv_thresh = ' + str(conv_thresh) + ": " + str(red_p[len(red_p)-1]) + ", " + str(blue_p[len(blue_p)-1]) + ", " + str(green_p[len(green_p)-1]))
                rloss_a = np.average(rloss_avg)
                rgain_a = np.average(rgain_avg)
                bloss_a = np.average(bloss_avg)
                bgain_a = np.average(bgain_avg)
                gloss_a = np.average(gloss_avg)
                ggain_a = np.average(ggain_avg)
                print('Red Loss (Average, %) at T=100%: ' + str(rloss_a))
                print('Red Gain (Average, %) at T=100%: ' + str(rgain_a))
                print('Blue Loss (Average, %) at T=100%: ' + str(bloss_a))
                print('Blue Gain (Average, %) at T=100%: ' + str(bgain_a))
                print('Green Loss (Average, %) at T=100%: ' + str(gloss_a))
                print('Green Gain (Average, %) at T=100%: ' + str(ggain_a))
                plt.title('Degree Distribution of Switching Nodes at T=100% Across All Seeds\n N=6000, rho_col=' + str(rho_col) + ', rho_res=' + str(rho_res) + ', rho_col=' + str(rho_col) + ', init_ratio=' + str((p_r,p_b,p_g)))
                plt.hist(overall_deg_dist)
                plt.xlabel('Degree')
                plt.ylabel('Count')
                plt.show()
                '''
            
            plt.rcParams.update({'font.size': 11})
            plt.plot(init_diffs, red_p,color='red')
            plt.plot(init_diffs, blue_p,color='blue')
            plt.plot(init_diffs, green_p,color='green')
            plt.xlabel(x_lab)
            plt.ylabel(y_lab)
            plt.title(title_str)
            plt.show()
            #plt.savefig(fname)
            plt.clf()
            

if __name__ == '__main__':
    main()