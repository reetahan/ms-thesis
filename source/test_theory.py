import numpy as np
import networkx as nx
import sys
import copy
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

def conversions(G,red,blue,conv_thresh, color_map):
    new_color_map = copy.deepcopy(color_map)
    did_conv = True

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
       
        print('============')
        print('Node: ' + str(n))
        print('Color: ' + str(cur_color))
        print('Threshold: ' + str(conv_thresh))
        rf = np.count_nonzero(neighbor_colors == 'red')/len(neighbor_colors)
        bf = np.count_nonzero(neighbor_colors == 'blue')/len(neighbor_colors)
        print('Red: ' + str(rd3(rf)) + " | Blue: " + str(rd3(bf)))
        if(rf > bf):
            if(cur_color == 'red'):
                if(rf == 1.0):
                    print('This will definitely stay red')
                else:
                    print('Stay 100% Red: (' + str(rd3(bf)) + ', 1]')
                    print('Stay ' + str(rd3(rf*100)) + '% Red: [0,' + str(rd3(bf)) + ']')
            else:
                if(rf == 1.0):
                    print('This will definitely turn red')
                else:
                    print('Convert To 100% Red: (' + str(rd3(bf)) + ', ' + str(rd3(rf)))
                    print('Convert To ' + str(rd3(rf*100)) + '% Red: [0,' + str(rd3(bf)) + ']')
        elif(rf < bf):
            if(cur_color == 'red'):
                if(bf == 1.0):
                    print('This will definitely turn blue')
                else:
                    print('Stay 100% Red: (' + str(rd3(bf)) + ', 1]')
                    print('Stay ' + str(rd3(rf*100)) + '% Red: [0,' + str(rd3(rf)) + ']')
            else:
                if(bf == 1.0):
                    print('This will definitely stay blue')
                else:
                    print('Convert To ' + str(rd3(rf*100)) + '% Red: [0,' + str(rd3(bf)) + ']')
        else:
            if(cur_color == 'red'):
                print('Stay 100% Red: (0.5,1]' )
                print('Stay 50% Red: [0,0.5]' )
            else:
                print('Convert To 50% Red: [0,0.5]')
        print('============')
        passing_color = []
        passing_values = []
        for c in ["red","blue"]:
            color_ct = np.count_nonzero(neighbor_colors == c)
            if(color_ct >= t):
                passing_color.append(c)
                passing_values.append(color_ct)

        selection = None
        if(len(passing_color) == 2 or (len(passing_color) == 1 and passing_color[0] != cur_color)):
            did_conv = False

        if(len(passing_color) > 0):
            probs = [x/sum(passing_values) for x in passing_values]
            selection = np.random.choice(passing_color, p=probs)


        if(selection == 'red'):
            my_color.remove(n)
            new_color_map[n] = 'red'
            red.add(n)

        if(selection == 'blue'):
            my_color.remove(n)
            new_color_map[n] = 'blue'
            blue.add(n)



    return red,blue,new_color_map, did_conv

def rd3(num):
    return round(num,3)

def main():

    G,red,blue, color_map = true_bpa_two(12,0.6,0.4,0.8)
    iteration = 0

    while(True):
        plt.title("Iteration: i = " + str(iteration) + " | R: " + str(rd3(len(red)/len(G.nodes))) + " B: " + str(rd3(len(blue)/len(G.nodes))))
        nx.draw(G,node_color=color_map)
        plt.show()
        plt.clf()
        
        red,blue,new_color_map, did_conv = conversions(G,red,blue,0.5,color_map)
        iteration = iteration + 1
        
        if(did_conv):
            plt.title("[CONVERGED!!!!] Iteration: i = " + str(iteration) + " | R: " + str(rd3(len(red)/len(G.nodes))) + " B: " + str(rd3(len(blue)/len(G.nodes))))
            nx.draw(G,node_color=new_color_map)
            plt.show()
            plt.clf()
            break
        color_map = new_color_map

        

    

if __name__ == '__main__':
    main()