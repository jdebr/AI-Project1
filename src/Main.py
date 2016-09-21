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

#Population array containing chromosome number as cell value
population = defaultdict(list)

#chromosome containing node as index value and color as cell value. It can be a dictionary too.
chromosome = []



def generate_points(n):
    ''' Generates n sets of points randomly scattered on the 
    unit square and stores them as tuples mapped to integer IDs
    '''
    coords.clear()
    
    for i in range(n):
        randX = random.randint(1, 1000000)
        randY = random.randint(1, 1000000)
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
    #This needs to be asked during graph point generation
    #numberOfColors = 4
    
    for i in range(numberOfColors):
        listOfColor.append(i)
    numberOfVertices = len(adjacent_matrix)
    tempColorList = copy.deepcopy(listOfColor)
    nodeNumber = 0
    
    print("Temp  Color List 1 " + str(tempColorList))
    while nodeNumber < len(adjacent_matrix):
        print("Node selected " + str(nodeNumber))
        
        if nodeNumber == 0 and not tempColorList:
            print("Game Over!!")
            break
                 
        for colors in range(len(listOfColor)):
            randomColor = tempColorList[random.randrange(len(tempColorList))]
            
            print("Color selected for assigning is " + str(randomColor))
            
            if checkAndAssignColor(nodeNumber, numberOfVertices, randomColor):
                colorNode[nodeNumber].append(randomColor)
                print("Testing of Main ColorNode " + str(colorNode.items()))
                tempColorList = copy.deepcopy(listOfColor)
                print("Temp Color List re-copy in case half empty and if not still, if all is well " + str(tempColorList))
                nodeNumber = nodeNumber + 1
                break
            
            #Brain of Back track
            else:
                tempColorList.remove(randomColor)
                print("Temp Color List in the first else block after removing color " + str(tempColorList))
                
                if colors + 1 == len(listOfColor) or not tempColorList:
                    
                    
                    print("Back Track Enter")
                    
                    nodeNumber = nodeNumber - 1
                    #To get the color Number of previous Node                   
                    colorToDelete = colorNode.__getitem__(nodeNumber)
                    print("Current Node Working and Popping from main " + str(nodeNumber))
                    #Converting to int for popping and Removing the brackets from color number
                    colorToDelete = str(colorToDelete).replace('[', '').replace(']', '')                    
                    colorInt = int(colorToDelete)
                    print("Color to delete is " + str(colorInt))
                    colorNode.pop(nodeNumber)
                    print("Main color node after popping " + str(colorNode.items()))
                    
                    if colorInt not in colorNodeChecker[nodeNumber] or nodeNumber not in colorNodeChecker:
                        
                        colorNodeChecker[nodeNumber].append(colorInt)
                        print("Color Node checker after appending and in if block " + str(colorNodeChecker.items()))
                        tempColorList = copy.deepcopy(listOfColor)
                        for values in colorNodeChecker[nodeNumber]:
                            print("Temp Color List before popping " + str(tempColorList))
                            print("Color values " + str(values))
                            tempColorList.remove(values)
                            
                    else:
                        print("Color Node checker in else block" + str(colorNodeChecker.items()))
                        tempColorList = copy.deepcopy(listOfColor)
                        for values in colorNodeChecker[nodeNumber]:
                            print("Temp Color List in else before popping " + str(tempColorList))
                            print(values)
                            tempColorList.remove(values)
                            
                            
                    if not tempColorList and len(colorNodeChecker[nodeNumber]) == numberOfColors:
                        print("Special case start")
                        colorNodeChecker.pop(nodeNumber) 
                        nodeNumber = nodeNumber - 1                       
                        tempColorList = copy.deepcopy(listOfColor)
                        specialColorToDelete = colorNode.__getitem__(nodeNumber)
                        print("Current Node Working and Popping " + str(nodeNumber))
                        #Converting to int for popping and Removing the brackets from color number
                        specialColorToDelete = str(specialColorToDelete).replace('[', '').replace(']', '')                    
                        specialColorInt = int(specialColorToDelete)
                        print("Color to delete is " + str(specialColorInt))
                        colorNode.pop(nodeNumber)
                        tempColorList.remove(specialColorInt)
                        if colorInt not in colorNodeChecker[nodeNumber]:
                            colorNodeChecker[nodeNumber].append(specialColorInt)
                    
                        #colorNode.pop(nodeNumber)
                        print("Main color node after popping " + str(colorNode.items()))
                        print("Special case end")
                                           
                    print("<><><><><><><> after popping " + str(tempColorList))
                    print("Node before exiting" + str(nodeNumber))
                    break
                
                elif nodeNumber == 0 and not tempColorList:
                    print("No more Possible Solutions")
                    break
                    
            
    print ("Final List ending backtrack " + str(colorNode.items()))                
   
def checkAndAssignColor(nodeNumber,totalVertices, colorNumber):
    for i in range(totalVertices):
      
        if adjacent_matrix[nodeNumber][i] == 1:
            if colorNumber in colorNode[i]:
                return False
        
    return True


#Genetic Algorithm Begins
def populationCreation(totalColor, noOfChromosome):
    
    
    for i in range(totalColor):
        listOfColor.append(i)
        
    for key in range(noOfChromosome):
        if not chromosome:
            for i in range(len(coords.items())):
                chromosome.append(listOfColor[random.randrange(len(listOfColor))])   
        
        else:
            chromosome[:] = []
            for i in range(len(coords.items())):
                chromosome.append(listOfColor[random.randrange(len(listOfColor))])
        print("Chromosome is ")
        print(chromosome)
        tempchromosome = copy.deepcopy(chromosome)
        population[key].append(tempchromosome)
 
    print("Final Population is ")
    print(population)
    

def modified_backtracking(numColors, backtrack_type = "simple"):
    ''' Initial call for recursive backtracking algorithm, takes number of colors we are 
    using and type ("simple," "forward", "mac") as parameters.
    Returns True if coloring is successful, else returns False
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
    return recursive_backtracking(numColors, backtrack_type)    


def recursive_backtracking(numColors, backtrack_type):
    ''' The recursive backtracking algorithm from Russell & Norvig pg 219.
    Takes number of colors and type of backtrack algorithm as parameters.
    Called automatically by modified_backtracking() and itself.
    Returns True if coloring is successful, else returns False
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
            inf = True
            if backtrack_type == "forward":
                inf = forward_check(currentNode, color)
            elif backtrack_type == "mac":
                inf = mac()
            # If inferences do not result in a failed coloring
            if inf:
                # Recursive call
                result = recursive_backtracking(numColors, backtrack_type)
                if result:
                    return result
        # We backtracked if we reach here so remove that color assignment
        colorNode[currentNode] = []    
        
    return False


def forward_check(nodeID, color):
    ''' Removes a recently assigned color from the domains of all unassigned
    adjacent nodes.  If this leaves any domains empty, undoes this process and 
    returns false.  Else returns true.
    '''
    altered = [] # local list to track node IDs for domains we change
    
    # Check all nodes for adjacency to current node and no color assignment
    for i in range(len(graph)):
        if adjacent_matrix[nodeID][i] == 1 and not colorNode[i]:
            # Remove from those nodes' domains the current color value
            if color in domains[i]:
                domains[i].remove(color)
                altered.append(i)
                
    # Check for empty domains
    for i in range(len(graph)):
        if not domains[i]:
            # If domain is empty, undo our domain change and return false
            print("Empty domain! Adding colors back")
            for j in altered:
                domains[j].append(color)
            return False
        
    # Otherwise keep domain changes and return true
    return True


def mac():
    return True


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
                #print("Adjacent nodes have same color!")
                #print("Node " + str(node1) + ": " + str(colorNode[node1]) + ", Node " + str(node2)) + ": " + str(colorNode[node2])
                tests_passed = False
                
    if tests_passed:
        print("Unit tests successfully passed!")
        print("Total Operations: ")
        print(OP_COUNT)
    else:
        print("Unit tests failed!")
        print("Total Operations: ")
        print(OP_COUNT)
        
    OP_COUNT = 0
    
            
def run_experiment_simple_backtracking(num_colors):
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
        # Create Adjacency Matrix
        matrix_creation()
        #print("Adj Matrix:")
        #print(adjacent_matrix.items())
        
        # Run Simple Backtracking
        print("##############################################################")
        print("Running Simple Backtracking  - " + str(num_colors) + " colors, "+ str(num_points) + " points")
        #BackTracking(4)
        print(get_time())
        print(modified_backtracking(num_colors, "simple")) 
        print(get_time())
        
        unit_tests()
        
        
def run_experiment_backtracking_forward_checking(num_colors):
    for i in range(1, 11):
        num_points = 10
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
        # Create Adjacency Matrix
        matrix_creation()
        #print("Adj Matrix:")
        #print(adjacent_matrix.items())
        
        # Run Backtracking w/ FC
        print("##############################################################")
        print("Running Backtracking w/ Forward Checking - " + str(num_colors) + " colors, "+ str(num_points) + " points")
        #BackTracking(4)
        print(get_time())
        print(modified_backtracking(num_colors, "forward")) 
        print(get_time())
        
        unit_tests()
        
        
def run_experiment_backtracking_MAC(num_colors):
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
        # Create Adjacency Matrix
        matrix_creation()
        #print("Adj Matrix:")
        #print(adjacent_matrix.items())
        
        # Run Backtracking w/ MAC
        print("##############################################################")
        print("Running Backtracking w/ MAC - " + str(num_colors) + " colors, "+ str(num_points) + " points")
        #BackTracking(4)
        print(get_time())
        print(modified_backtracking(num_colors, "mac")) 
        print(get_time())
        
        unit_tests()
       
       
def main():
    #run_experiment_simple_backtracking(3)
    #run_experiment_simple_backtracking(4)
    #run_experiment_backtracking_forward_checking(3)
    run_experiment_backtracking_forward_checking(4)
    #run_experiment_backtracking_MAC(3)
    #run_experiment_backtracking_MAC(4)
    pass
    
if __name__ == '__main__':
    main()
