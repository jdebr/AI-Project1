'''
CSCI 446 
Fall 2016
Project 1

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

import random
import math
from collections import defaultdict
import matplotlib.pyplot as py
from networkx.classes.function import is_empty
from Assignment import colorList



graph = defaultdict(list)
# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}


color = {}
# Maps node ID to some color value
# {nodeID: 'color'}

colorList = []
#Maps vertex number to key value


#Adjacency Matrix
adjacent_matrix = defaultdict(list)

coords = {}
# Maps node ID to a tuple representing Cartesian coordinates
# {nodeID: (x, y)}


distance = defaultdict(list)
# Maps node ID to a list of tuples which represent NodeID and distance 
# {nodeID: [(nodeID, distance)...]}  


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
                                    


def matrix_creation():
    numberofvertices = 5
    available_nodes_coloring = list(graph.keys())
    print("Color Available node " + str(available_nodes_coloring))
    
    '''
    this can go in another function or maybe used by other algorithms.
    This is the block where adjacency matrix is made
    '''
    while available_nodes_coloring:
        random_point = random.choice(available_nodes_coloring)
        '''
        for testing
        print(random_point)
        '''
        for i in range(numberofvertices):
            if i == random_point:
                adjacent_matrix[random_point].append(1)
            elif i not in graph[random_point] and random_point not in graph[i]:
                adjacent_matrix[random_point].append(0)
            else:
                adjacent_matrix[random_point].append(1)
        available_nodes_coloring.remove(random_point)   
    print(adjacent_matrix.items())
    BackTracking()
    

def BackTracking():
    numberOfColors = 4
    colorValue = 0
    numberOfVertices = len(adjacent_matrix)
    nodeNumber = 0
    while colorValue < numberOfColors:
        if nodeNumber < numberOfVertices:  
            if checkAndAssignColor(nodeNumber,numberOfVertices,colorValue):
                colorList.append(colorValue)
                print("Initial Color List")
                print(colorList)
                colorValue = 0
                nodeNumber = nodeNumber + 1
            else:
                print("Increasing Color Value")
                colorValue = colorValue + 1
        else:
            break
        
    
    print(colorList)
            
                
   
def checkAndAssignColor(nodeNumber,totalVertices, colorNumber):
    for i in range(totalVertices):
        
        '''Testing
        print("///////////////////////")
        print(adjacent_matrix[nodeNumber][i])
        print("///////////////////////")
        '''
        if adjacent_matrix[nodeNumber][i] == 1:
            if not colorList:
                return True
            else:
                lengthOfColorArray = len(colorList)
                print(lengthOfColorArray)
                if i >= lengthOfColorArray:
                    pass
                else:
                    if colorNumber == colorList[i]:
                        return False
                    
                
    return True
                              
def plot_graph():
    print(coords)
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
    matrix_creation()
    plot_graph()
    
            

    
        
def main():
    run_experiment()
    #unit_tests()
    
    
if __name__ == '__main__':
    main()
