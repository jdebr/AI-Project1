import random
import math
import operator
import time 
from collections import defaultdict
from sortedcontainers import SortedDict
import matplotlib.pyplot as py

graph = defaultdict(list)
# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}

color = {}
# Maps node ID to some color value
# {nodeID: 'color'}

coords = {}
# Maps node ID to a tuple representing Cartesian coordinates
# {nodeID: (x, y)}

distance = defaultdict(list)
# Maps node ID to a list of tuples which represent NodeID and distance 
# {nodeID: [(nodeID, distance)...]}  

conflicts = {}
#List of the number of conflic for each vertex 

def generate_points(n):
    ''' Generates n sets of points randomly scattered on the 
    unit square and stores them as tuples mapped to integer IDs
    '''
    for i in range(n):
        randX = random.randint(1, 10000)
        randY = random.randint(1, 10000)
        v = (float(randX), float(randY))
        coords[i] = v 
        

def calculate_distances():
    ''' Calculates distances between each pair of points,
    storing as a dictionary of lists of tuples, then sorting
    the lists by distance
    '''
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # Calculate distance from i to j, 
            # store in both distances[i] and distances[j]
            delta_x = coords[i][0] - coords[j][0]
            delta_y = coords[i][1] - coords[j][1]
            d = math.sqrt(pow(delta_x, 2) + pow(delta_y, 2))
            distance[i].append((j, d))
            distance[j].append((i, d)) 
            
    # Sort each list of tuples by distance
    for k, v in distance.items():
        v.sort(key=lambda d: d[1])
        
        
def build_graph():
    ''' Add edges to graph by selecting a random point and adding 
    an edge to the nearest point that doesn't already have an edge
    and such that it will not cross any other edge'''
    lines = []
    # List of existing line segments (edges on graph)
    # [(A, B, C, x1, y1, x2, y2) ...]
    
    available_nodes = list(coords.keys())
    # List of node IDs to check for possible edges
    
    
    # Start main loop - run until no connections available
    while available_nodes:
        # Choose random point from available nodes
        selected_point = random.choice(available_nodes)
        # Boolean to loop until a line is added
        lineAdded = False
        while not lineAdded:
            # If there's no nodes left to check for selected node, remove from 
            # available nodes and exit loop to select a new node
            if not distance[selected_point] :
                available_nodes.remove(selected_point)
                break
            # Get closest point to our chosen point and remove it from dictionary
            # to indicate we have tried to add an edge to it.  
            else : 
                nearest_point = distance[selected_point][0][0]
                distance[selected_point].pop(0)
                
                # if no edge already exists between these points, calculate line equation
                if nearest_point not in graph[selected_point] and selected_point not in graph[nearest_point]:
                    # set coordinates for calculation
                    pt1 = coords[selected_point]
                    pt2 = coords[nearest_point]
                    x1 = pt1[0]
                    y1 = pt1[1]
                    x2 = pt2[0]
                    y2 = pt2[1]
                    
                    #Check if vertical line, might need to handle differently?
                    if x1 == x2:
                        print("Vertical line!")
                        pass
                    # Check if this line will intersect any other edges in graph
                    else:
                        # Calculate values for line Ax + By = C
                        A1 = y2 - y1
                        B1 = x1 - x2
                        C1 = A1 * x1 + B1 * y1
                        # If no edges on graph then no conflict, so add this as first edge
                        if not lines:
                            graph[selected_point].append(nearest_point)
                            lines.append((A1, B1, C1, x1, y1, x2, y2))
                            lineAdded = True
                        # Check all other edges for potential intersection
                        else:
                            intersection_found = False
                            
                            for line in lines:
                                # Retrieve pre-calculated line information
                                A2 = line[0]
                                B2 = line[1]
                                C2 = line[2]
                                x3 = line[3]
                                y3 = line[4]
                                x4 = line[5]
                                y4 = line[6]
                                
                                # Calculate determinant of system
                                det = (A1 * B2) - (A2 * B1)
                                if det == 0:
                                    # edges are parallel
                                    if abs(A1) == abs(A2) and abs(B1) == abs(B2) and abs(C1) == abs(C2):
                                        # edges fall on same line, so check for overlap
                                        if x1 == x3 or x1 == x4 and x2 == x3 or x2 == x4:
                                            # edges are exactly the same
                                            print("Same Line!")
                                            intersection_found = True
                                        elif (x1 > min(x3, x4) and x1 < max(x3, x4)
                                                or x2 > min(x3, x4) and x2 < max(x3, x4)
                                                or x3 > min(x1, x2) and x3 < max(x1, x2)
                                                or x4 > min(x1, x2) and x4 < max(x1, x2)):
                                            # Parallel edges overlap and thus intersect
                                            print("Parallel Overlap!")
                                            intersection_found = True
                                        else:
                                            # Parallel lines don't overlap so we are ok
                                            pass
                                # Calculate point of intersection
                                else:
                                    x = (B2 * C1 - B1 * C2) / det
                                    y = (A1 * C2 - A2 * C1) / det
                                    
                                    # Check if on segment 1
                                    if x > min(x1, x2) and x < max(x1, x2):
                                        if y > min(y1, y2) and y < max(y1, y2):
                                            # Check if on segment 2
                                            if x > min(x3, x4) and x < max(x3, x4):
                                                if y > min(y3, y4) and y < max(y3, y4):
                                                    # this will intersect so move on
                                                    intersection_found = True
                                                    break
                                                
                            # No intersection so add edge and line
                            if not intersection_found:
                                graph[selected_point].append(nearest_point)
                                lines.append((A1, B1, C1, x1, y1, x2, y2))
                                lineAdded = True
                                    
                                
def plot_graph():
    xval = []
    yval = []
    for i in range(len(coords)):
        xval.append(coords[i][0])
        yval.append(coords[i][1])
    py.plot(xval, yval, 'or')   
    
    for pt, edges in graph.items():
        for pt2 in edges:
            xval = [coords[pt][0], coords[pt2][0]]
            yval = [coords[pt][1], coords[pt2][1]]
            py.plot(xval, yval)
    
    py.show()
    
def creat_adgacent_matrix():   
    #init adjacent matrix
    matrix_adj = list()
    for x in range(0, len(graph)):
        raw = list()
        for y in range(0,len(graph)):
            raw.append(0)
        matrix_adj.append(raw)
    #Creat matrix
    for i in range(0,len(graph)) :
        for j in range(0, len(graph)):
            if j in graph[i] : 
                matrix_adj[i][j] = 1
                matrix_adj[j][i] = 1
    return matrix_adj
 
def random_color(nb):
    return random.randint(0, nb-1)
    
def init_graph_color(nb):
    for v in graph : 
        color[v] = random_color(nb)

#compute the number of conflicts for a vertex v and save it in the list conflicts
def nb_conflicts(v, mat_adj) : 
    nb = 0
    print "raw for vertex " + str(v) + " is : " + str(mat_adj[v])
    for pts in mat_adj[v]:
        if mat_adj[v][pts] == 1 and color[v] == color[pts] : 
            nb= nb + 1
    conflicts[v] = nb

#sum of all the conflicts 	
def tot_conflicts(mat_adj):
    for v in graph :
        nb_conflicts(v,mat_adj)
    return sum(conflicts.itervalues())

	
def test_csp(mat_adj): 
    if tot_conflicts(mat_adj) == 0 :
        return True
    else :
        return False

def minimize_conflicts(mat_adj, nb):
    print "premiere matrix de col"
    print color
    nb_tot_conf = tot_conflicts(mat_adj)
    print "nb conf init = " + str(nb_tot_conf)
    list_conf = conflicts
    max_conf = max(conflicts.iteritems(), key=operator.itemgetter(1))[0]
    print "conflicts " + str(conflicts)
    print "max_conf " + str(max_conf)
    col_max = color[max_conf]
    new_col = random_color(nb)
    while col_max == new_col :
        new_col = random_color(nb)
    color[max_conf] = new_col
    print "deuxieme matrix de col"
    print color
    new_nb_tot_conf = tot_conflicts(mat_adj)
    print "nb_tot_conf est " + str(nb_tot_conf) + "    new_nb_tot_conf est " + str(new_nb_tot_conf)
    while new_nb_tot_conf >= nb_tot_conf or conflicts[max(list_conf.iteritems(), key=operator.itemgetter(1))[0]] != 0:
        time.sleep(5)
        color[max_conf] = col_max
        list_conf[max_conf] = 0
        print "list_conf = " + str(list_conf)
        max_conf = max(list_conf.iteritems(), key=operator.itemgetter(1))[0]
        print "max_conf " + str(max_conf)
        print "color" + str(color)
        print color[max_conf]
        col_max = color[max_conf]
        new_col = random_color(nb)
        while col_max == new_col :
            new_col = random_color(nb)
        color[max_conf] = new_col
        new_nb_tot_conf = tot_conflicts(mat_adj)
    print "nb conf fin = " + str(new_nb_tot_conf)
        
    
def min_conflicts(max_it,nb) : 
    mat_adj = creat_adgacent_matrix()
    tot_conf = tot_conflicts(mat_adj)
    print "tot_conf is " + str(tot_conf)
    init_graph_color(nb)
    print color
    if not test_csp : 
        for i in range(1, max_it) :
            minimize_conflicts(mat_adj, nb)
            if test_csp(mat_adj):
                print "Vrai"
                return True
        print "Failure"
        return False
    else :
        print "Vrai"
        return True		

    
    
def unit_tests():
    for i in range(1000):
        generate_points(3)
        calculate_distances()
        build_graph()
        x = 0
        for node, edges in graph.items():
            x += len(edges)
        
        if x != 3:
            print("Num Edges: " + str(x))
            print("Error")
            
            
def run_experiment():
    generate_points(5)
    #for key, value in coords.items():
    #    print(key, value)  
    calculate_distances()
    #print(distance)
    
    build_graph()
    #for key, value in graph.items():
    #print("Node " + str(key) + " connected to nodes: " + str(value))
    
    #plot_graph()
    
    mat = creat_adgacent_matrix ()
    min_conflicts(100,4)
    print mat
    
        
def main():
    run_experiment()
    #unit_tests()
    
    
if __name__ == '__main__':
    main()