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
import copy
import datetime



graph = defaultdict(list)
# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}


colorNode = defaultdict(list)
# Maps node ID to some color value
# {nodeID: 'color'}


colorNodeChecker = defaultdict(list)
# Maps node ID to some color value
# {nodeID: 'color'}


colorList = []
listOfColor = []
#Maps vertex number to key value


adjacent_matrix = defaultdict(list)
#Adjacency Matrix


coords = {}
# Maps node ID to a tuple representing Cartesian coordinates
# {nodeID: (x, y)}


distance = defaultdict(list)
# Maps node ID to a list of tuples which represent NodeID and distance 
# {nodeID: [(nodeID, distance)...]}  


domains = defaultdict(list)
# Maps node ID to a list of color values that are legal for that node
# {nodeID: [color1, color2, ...]}


OP_COUNT = 0
# Counter to track number of times a node is assigned a color within an algorithm


def generate_points(n):
    ''' Generates n sets of points randomly scattered on the 
    unit square and stores them as tuples mapped to integer IDs
    '''
    coords.clear()
    
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
    distance.clear()
    
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
    graph.clear()
    
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
    '''This is the block where adjacency matrix is made'''
    adjacent_matrix.clear()
    
    available_nodes_coloring = list(graph.keys())
    #print("Node's to Color " + str(available_nodes_coloring))
        
    while available_nodes_coloring:
        random_point = random.choice(available_nodes_coloring)
        '''
        for testing
        print(random_point)
        '''
        for i in range(len(graph)):
            if i == random_point:
                adjacent_matrix[random_point].append(0)
            elif i not in graph[random_point] and random_point not in graph[i]:
                adjacent_matrix[random_point].append(0)
            else:
                adjacent_matrix[random_point].append(1)
        available_nodes_coloring.remove(random_point)   
    

def BackTracking(numberOfColors):
    ''' Simple Backtracking Algorithm Start '''
    for i in range(numberOfColors):
        listOfColor.append(i)
        
    numberOfVertices = len(adjacent_matrix)
    tempColorList = copy.deepcopy(listOfColor)
    nodeNumber = 0
    #print("Temp Color List " + str(tempColorList))
    while nodeNumber < len(adjacent_matrix):
        #print("Node selected " + str(nodeNumber))
        for colors in range(len(listOfColor)):
            randomColor = tempColorList[random.randrange(len(tempColorList))]
            #print("Color selected is " + str(randomColor))
            if checkAndAssignColor(nodeNumber, numberOfVertices, randomColor):
                colorNode[nodeNumber].append(randomColor)
                #print("Temp Color List 2 " + str(tempColorList))
                #print("Test this " + str(colorNode.items()))
                tempColorList = copy.deepcopy(listOfColor)
                #print("Temp Color List 3 " + str(tempColorList))
                nodeNumber = nodeNumber + 1
                break

            else:
                tempColorList.remove(randomColor)
                #print("Temp Color List 4 " + str(tempColorList))                
                if colors + 1 == len(listOfColor) or not tempColorList:
                    #print("Back Track")
                    nodeNumber = nodeNumber - 1                   
                    colorToDelete = colorNode.__getitem__(nodeNumber)
                    #print("Current Node Working " + str(nodeNumber))
                    
                    colorToDelete = str(colorToDelete).replace('[', '').replace(']', '')
                    
                    colorInt = int(colorToDelete)
                    #print(colorInt)
                    
                    colorNode.pop(nodeNumber)
                    if colorInt not in colorNodeChecker[nodeNumber]:
                        colorNodeChecker[nodeNumber].append(colorInt)
                        #print("Color Node checker " + str(colorNodeChecker.items()))
                        for key, values in colorNodeChecker.items():
                            tempColorList = copy.deepcopy(listOfColor)
                            for values in colorNodeChecker[nodeNumber]:
                                #print("Temp Color List 5 " + str(tempColorList))
                                #print(values)
                                tempColorList.remove(values)
                    else:
                        #print("Color Node checker " + str(colorNodeChecker.items()))
                        for key, values in colorNodeChecker.items():
                            tempColorList = copy.deepcopy(listOfColor)
                            for values in colorNodeChecker[nodeNumber]:
                                #print("Temp Color List 6 " + str(tempColorList))
                                #print(values)
                                tempColorList.remove(values)
                                            
                    #print("<><><><><><><>" + str(tempColorList))
                    #print("Node" + str(nodeNumber))
                    break
                               
   
def checkAndAssignColor(nodeNumber,totalVertices, colorNumber):
    ''' For node ID nodeNumber and the number of vertices in the graph totalVertices,
    returns TRUE if no adjacent vertices already have the color represented by colorNumber,
    and FALSE otherwise
    '''
    for i in range(totalVertices):
        if adjacent_matrix[nodeNumber][i] == 1:
            if colorNumber in colorNode[i]:
                return False
        
    return True 
'''
        OldMethod
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
                        '''    


def modified_backtracking(numColors):
    ''' Initial call for recursive backtracking algorithm
    Returns True if coloring is successful,
    else returns False
    '''
    # INITIALIZATION OF GLOBALS
    # Clear color nodes
    colorNode.clear()
    # Initialize colorNode dictionary for coloring_complete() method
    for i in range(len(graph)):
        colorNode[i] = []
    # Initialize domain values
    initialize_domains(numColors)
    
    # Begin recursion
    return recursive_backtracking(numColors)    


def recursive_backtracking(numColors):
    ''' The recursive backtracking algorithm from Russell & Norvig pg 219.
    Returns True if coloring is successful,
    else returns False
    '''
    # Base Case
    if coloring_complete():
        print("Coloring Completed!")
        return True
    # Select Unassigned Variable, use MRV heuristic?
    currentNode = select_mrv()
    # Iterate through colors available for currentNode
    for color in domains[currentNode]:
        if checkAndAssignColor(currentNode, len(graph), color):
            colorNode[currentNode].append(color)
            # Track number of node colorings
            incr_op_count()
            # INFERENCE STEP HERE
            # Recursive call
            result = recursive_backtracking(numColors)
            if result:
                return result
        # We backtracked if we reach here so remove that color assignment
        colorNode[currentNode] = []    
        
    return False


def select_mrv():
    ''' Select an uncolored node with the smallest domain, 
    return its ID
    '''
    selectedID = -1
    # Start with an uncolored node with a nonzero domain
    for i in range(len(graph)):
        if not colorNode[i] and len(domains[i]) > 0:
            selectedID = i
    
    # Check all uncolored nodes for a smaller nonzero domain
    for node, domain in domains.items():
        if not colorNode[node] and len(domain) > 0:
            if len(domain) < len(domains[selectedID]):
                selectedID = node
                
    return selectedID


def coloring_complete():
    ''' Returns true if all nodes have been assigned a color in colorNodes,
    else returns false
    '''
    for nodes, colors in colorNode.items():
        if not colors:
            return False
    return True


def initialize_domains(numColors):
    ''' Initializes the color domains for each node in the graph '''
    domains.clear()
    for i in range(len(graph)):
        for j in range(numColors):
            domains[i].append(j)
                   
                              
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
    
    
def get_time():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')


def incr_op_count():
    global OP_COUNT
    OP_COUNT = OP_COUNT + 1
    
    
def unit_tests():
    tests_passed = True
    global OP_COUNT
    
    # Graph creation test
    '''
    for i in range(1000):
        generate_points(3)
        calculate_distances()
        build_graph()
        x = 0
        for node, edges in graph.items():
            x += len(edges)
        
        if x != 3:
            print("Num Edges: " + str(x))
            tests_passed = False
    '''
    
    # Successful coloring test
    for node1, v in graph.items():
        for node2 in v:
            if colorNode[node1] == colorNode[node2]:
                print("Adjacent nodes have same color!")
                print("Node " + str(node1) + ": " + str(colorNode[node1]) + ", Node " + str(node2)) + ": " + str(colorNode[node2])
                tests_passed = False
                
    if tests_passed:
        print("Unit tests successfully passed!")
        print("Total Operations: ")
        print(OP_COUNT)
    OP_COUNT = 0
    
            
def run_experiment():
    for i in range(1, 11):
        num_points = i * 10
        # Scatter Points
        generate_points(num_points)
        #print("Coordinates:")
        #print(coords.items())
        
        # Determine Euclidean Distances
        calculate_distances()    
        #print("Euclidean Distances")
        #print(distance.items())
        
        # Connect Edges
        build_graph()
        
        # Show Visual Plot
        #plot_graph()
        
        # Create Adjacency Matrix
        matrix_creation()
        #print("Adj Matrix:")
        #print(adjacent_matrix.items())
        
        # Run Simple Backtracking
        #print("Running Simple Backtracking")
        #BackTracking(4)
           
        
        # Run Backtracking w/ Forward Checking
        print("Running Backtracking w/ Forward Checking - " + str(num_points))
        print(get_time())
        print(modified_backtracking(4)) 
        print(get_time())
        #print("Color Assignments:")
        #print(colorNode.items())  
        
        unit_tests()
       
       
def main():
    run_experiment()
    #unit_tests()
    #plot_graph()
    
    
if __name__ == '__main__':
    main()
