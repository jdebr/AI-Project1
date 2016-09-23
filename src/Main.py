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
from collections import defaultdict, deque
import matplotlib.pyplot as py
import copy
import datetime
import operator



graph = defaultdict(list)
# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}


colorNode = defaultdict(list)
# Maps node ID to some color value
# {nodeID: 'color'}


color = {}
# Maps node ID to some color value
# {nodeID: 'color'}


conflicts = {}
#List of the number of conflic for each vertex 


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


population = defaultdict(list)
#Population array containing chromosome number as cell value


tempParent = defaultdict(list)
#Store the value of 2 random selected chromosome aka parent.

#Actual Parents which store fit chromosome
parent1 = []
parent2 = []

#fitness Dictionary
fitness = {}


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
        
    return adjacent_matrix
    
    
# START MIN CONFLICTS IMPORT
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
    for pts in range(len(graph)):
        '''
        print("-------------------")
        print(mat_adj[v][pts])
        print(color[v])
        print(color[pts])
        print("-------------------")
        '''
        if mat_adj[v][pts] == 1 and color[v] == color[pts] : 
            nb= nb + 1
    conflicts[v] = nb
    
def tot_conflicts(mat_adj):
    for v in graph :
        nb_conflicts(v,mat_adj)
    tot_conf = 0
    for v in conflicts : 
        tot_conf = tot_conf + conflicts[v]
    return tot_conf

    
def test_csp(mat_adj): 
    if tot_conflicts(mat_adj) == 0 :
        return True
    else :
        return False

def minimize_conflicts(mat_adj, nb):
    print("premiere matrix de col / initial color matrix")
    print(color)
    nb_tot_conf = tot_conflicts(mat_adj)
    list_conf = conflicts
    max_conf = max(conflicts.items(), key=operator.itemgetter(1))[0]
    print("conflicts " + str(conflicts))
    print("max_conf " + str(max_conf))
    col_max = color[max_conf]
    new_col = random_color(nb)
    while col_max == new_col :
        new_col = random_color(nb)
    color[max_conf] = new_col
    print("deuxieme matrix de col / new color matrix")
    print(color)
    new_nb_tot_conf = tot_conflicts(mat_adj)
    print("nb_tot_conf est " + str(nb_tot_conf) + "    new_nb_tot_conf est " + str(new_nb_tot_conf))
    while new_nb_tot_conf >= nb_tot_conf  or conflicts[max(list_conf.items(), key=operator.itemgetter(1))[0]] != 0:  
        list_conf[max_conf] = 0
        max_conf = max(conflicts.items(), key=operator.itemgetter(1))[0]
        print("*******************************************************")
        print("max_conf node # " + str(max_conf))
        print("color" + str(color))
        print("color of node " + str(max_conf) + ": " + str(color[max_conf]))
        col_max = color[max_conf]
        new_col = random_color(nb)
        while col_max == new_col :
            new_col = random_color(nb)
        color[max_conf] = new_col
        '''
        changes by Shriyansh
        '''
        #nb_tot_conf = new_nb_tot_conf
        new_nb_tot_conf = tot_conflicts(mat_adj)
        print("Old # conflicts: " + str(nb_tot_conf))
        print("New # conflicts: " + str(new_nb_tot_conf))
        print("Color is " + str(color))
        
    
def min_conflicts(max_it,nb) : 
    #mat_adj = matrix_creation()
    mat_adj = creat_adgacent_matrix()
    init_graph_color(nb)
    for i in range(1, max_it) :
        if test_csp(mat_adj):
            print("Vrai")
            return True
        minimize_conflicts(mat_adj, nb)
        
    print("Failure")
    return False

# END MIN CONFLICTS IMPORT
    
#Start of Non Recursive Simple Back Tracking 
def NonRecursiveSimpleBackTracking(numberOfColors):
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
                    colorToDelete = removeBracketsMakeInt(colorNode.__getitem__(nodeNumber))
                    print("Current Node Working and Popping from main " + str(nodeNumber))
                    #Converting to int for popping and Removing the brackets from color number
                    print("Color to delete is " + str(colorToDelete))
                    colorNode.pop(nodeNumber)
                    print("Main color node after popping " + str(colorNode.items()))
                    
                    if colorToDelete not in colorNodeChecker[nodeNumber] or nodeNumber not in colorNodeChecker:                       
                        colorNodeChecker[nodeNumber].append(colorToDelete)
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
                        specialColorToDelete = removeBracketsMakeInt(colorNode.__getitem__(nodeNumber))
                        print("Current Node Working and Popping " + str(nodeNumber))
                        #Converting to int for popping and Removing the brackets from color number
                        print("Color to delete is " + str(specialColorToDelete))
                        colorNode.pop(nodeNumber)
                        tempColorList.remove(specialColorToDelete)
                        if specialColorToDelete not in colorNodeChecker[nodeNumber] or nodeNumber not in colorNodeChecker:                       
                            colorNodeChecker[nodeNumber].append(specialColorToDelete)
                        print("Main color node after popping " + str(colorNode.items()))
                        print("Special case end")
                                           
                    print("<><><><><><><> after popping " + str(tempColorList))
                    print("Node before exiting" + str(nodeNumber))
                    break
                
                elif nodeNumber == 0 and not tempColorList:
                    print("No more Possible Solutions")
                    break
                    
            
    print ("Final List ending backtrack " + str(colorNode.items()))  
#End of Non-Recursive Simple Back Tracking 

#Start of Recursive Simple Back Tracking

def RecursiveSimpleBackTracking(numberOfColors,nodeNumber):
    if not (brainBackTracking(numberOfColors,nodeNumber)):
        print("Sorry No Solution")
        return False
    
    print(colorNode)
    return True
        
    
    
def brainBackTracking(numberOfColors, nodeNumber):
    for i in range(numberOfColors):
        if nodeNumber == len(adjacent_matrix):
            return True
        if(checkAndAssignColor(nodeNumber, len(adjacent_matrix), i)):
            colorNode[nodeNumber].append(i)
            
            if brainBackTracking(numberOfColors, nodeNumber + 1):
                return True
                            
            colorNode.pop(nodeNumber)
            
    return False

    print("final Colors")
    print(colorNode)
   
def checkAndAssignColor(nodeNumber,totalVertices, colorNumber):
    for i in range(totalVertices):
      
        if adjacent_matrix[nodeNumber][i] == 1:
            if colorNumber in colorNode[i]:
                return False
        
    return True
    
#End of Recursive Simple Back Tracking    
    
#Start of Genetic Algorithm
def genetic_algorithm(num_colors, max_iterations, population_size):
    ''' The genetic algorithm from Russell & Norvig, pg 130.
    Builds a population of randomized solutions to the graph 
    coloring problem, selects individuals via a pairwise tournament
    selection, then builds a new generation by pairing and crossing
    over the winners, until a solution is found or max_iterations is reached.
    '''
    populationCreation(num_colors, population_size)
    global population
    global OP_COUNT
    # Main algorithm loop
    for i in range(max_iterations):
        # Test for solution, return if it exists
        for j in range(population_size):
            if calculateFitness(population[j]) == 0:
                OP_COUNT = i
                return population[j]
            
        # If no solution, generate new population with GA
        newPopulation = {}
        # Loop to create new generation
        for j in range(population_size):
            # Select two individuals at random from population
            x = population[random.randint(0, len(population)-1)]
            y = population[random.randint(0, len(population)-1)]
            # Pairwise tournament selection advances the fittest to crossover as parent 1
            if calculateFitness(x) > calculateFitness(y):
                parent1 = y
            else:
                parent1 = x 
                
            # Select two individuals at random from population
            x = population[random.randint(0, len(population)-1)]
            y = population[random.randint(0, len(population)-1)]
            # Pairwise tournament selection advances the fittest to crossover as parent 2
            if calculateFitness(x) > calculateFitness(y):
                parent2 = y
            else:
                parent2 = x 
                
            child = reproduce(parent1, parent2)
            
            mutate(child, num_colors)
            
            newPopulation[j] = child
            
        population = newPopulation
    # If max iterations reached, return false
    return False        
        

def populationCreation(totalColor, noOfChromosome):
    ''' Initializes a population of size noOfChromosome, consisting of randomized 
    solutions to graph coloring problem, using a total number of colors totalColor.
    '''
    for i in range(totalColor):
        listOfColor.append(i)
        
    for key in range(noOfChromosome):
        chromosome = [random.randint(0, totalColor-1) for x in range(len(graph))]
        population[key] = chromosome
        

def calculateFitness(chromosome):
    ''' Calculates the fitness score of an individual.  Our fitness function is the number of 
    conflicts counted in the individual solution, which means our ideal fitness is 0 and higher
    fitness scores are worse.
    '''
    fitness = 0
    # Count number of conflicts in a solution.  A conflict is found if two nodes are adjacent in 
    # the graph and they also have the same color.
    for i in range(len(graph)):
        for j in range(len(graph)):
            if adjacent_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                fitness += 1
    return fitness


def reproduce(parent1, parent2):
    ''' Combines two parent solutions into one new solution by splitting
    each solution at a random index and appending the first [1 to index-1] values 
    from parent1 to the [index to n] values from parent 2
    '''
    index = random.randint(0, len(parent1)-1)
    child = parent1[:index]
    child[index:] = parent2[index:]
    return child


def mutate(child, num_colors):
    ''' Changes color values in the string representing a solution, child, according to
    some small probability.  Colors are selected from num_colors.  Returns child.
    '''
    p = 0.01
    for i in range(len(child)):
        r = random.random()
        if r < p:
            child[i] = random.randint(0,num_colors-1)
    return child
            
        

def removeBracketsMakeInt(toCovertValue):
        tempValue = toCovertValue
        tempValue = str(tempValue).replace('[', '').replace(']', '')                    
        tempValueInt = int(tempValue)
        return tempValueInt

#End of Genetic Algorithm
    

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
    # Local variable for iterating through domains
    currentDomain = copy.deepcopy(domains[currentNode])
    # Local variable for inference results
    inf = (True, [])
    # Iterate through colors available for currentNode
    for color in currentDomain:
        if checkAndAssignColor(currentNode, len(graph), color):
            # Color the node 
            colorNode[currentNode].append(color)
            # reduce the domain of that node to that color temporarily for MAC
            restoreColors = []
            for c in domains[currentNode]:
                if c != color:
                    restoreColors.append(c) 
            for c in restoreColors:
                domains[currentNode].remove(c)   
            # Track number of node colorings
            incr_op_count()
            # INFERENCE STEP HERE - inf will contain boolean representing inference success
            # and a dict of lists of altered node domains in case changes need to be reverted
            if backtrack_type == "forward":
                inf = forward_check(currentNode, color)
            elif backtrack_type == "mac":
                inf = mac(currentNode, color)
            # If inferences do not result in a failed coloring
            if inf[0]:
                # Recursive call
                result = recursive_backtracking(numColors, backtrack_type)
                if result:
                    return result
            # Restore domain of current node
            for c in restoreColors:
                domains[currentNode].append(c)
        # We backtracked if we reach here so remove that color assignment and undo inference changes
        colorNode[currentNode] = []
        if backtrack_type != "simple":
            for node, colors in inf[1].items():
                for c in colors:
                    domains[node].append(c)
            
        
    return False


def forward_check(nodeID, color):
    ''' Removes a recently assigned color from the domains of all unassigned
    adjacent nodes.  Returns a tuple containing a boolean and a list of nodeIDs
    for altered domains, where the boolean is false if some domain is empty and 
    true otherwise.
    '''
    altered = defaultdict(list) 
    #local var to track changed nodes, maps node ID to list of colors removed from that 
    #node's domain
    
    # Check all nodes for adjacency to current node and no color assignment
    for i in range(len(graph)):
        if adjacent_matrix[nodeID][i] == 1 and not colorNode[i]:
            # Remove from those nodes' domains the current color value
            if color in domains[i]:
                domains[i].remove(color)
                altered[i].append(color)
                
    # Check for empty domains
    for i in range(len(graph)):
        if not domains[i]:
            # If any domain is empty, return false
            return (False, altered)
        
    # Otherwise keep domain changes and return true
    return (True, altered)


def mac(nodeID, color):
    ''' An implementation of AC-3 for Arc Consistency from Russell & Norvig, pg 209.
    Used by inference step of recursive backtracking for the Maintaining Arc Consistency
    (MAC) inference for Recursive Backtracking.  
    Returns false if some domain is reduced to empty set, else true.
    '''
    altered = defaultdict(list) 
    #local var to track changed nodes, maps node ID to list of colors removed from that 
    #node's domain so we can replace them later if we backtrack
    
    # Initialize Queue of arcs to check
    arcQueue = deque()
    for i in range(len(graph)):
        if adjacent_matrix[nodeID][i] == 1 and not colorNode[i]:
            arcQueue.append((i, nodeID))
    
    # Loop until Queue is empty
    while arcQueue:
        arc = arcQueue.popleft()
        # Run revise method, which returns a tuple of a boolean along with the altered dictionary for undoing changes
        revised = revise(arc)
        if revised[0]:
            # populate altered dictionary for reversing domain changes later
            for n, c in revised[1].items():
                for col in c:
                    altered[n].append(col)
            # check for empty domain, if found return false
            if not domains[arc[0]]:
                return (False, altered)
            
            # add arcs to queue that are neighbors of arc[0], unassigned, and not nodeID
            for i in range(len(graph)):
                if adjacent_matrix[arc[0]][i] == 1 and not colorNode[i] and i != nodeID:
                    arcQueue.append((i, arc[0]))
                    
    return (True, altered)


def revise(arc):
    ''' Used by mac() (AC-3) to revise domains of nodes to enforce arc consistency.
    Checks domain of arc[0] for colors that can be consistent with each color in arc[1]'s
    domain, if not that color is removed from arc[0]'s domain.
    Returns a tuple: (bool, altered) where bool is true if domain is revised, else false. 
    '''
    altered = defaultdict(list) 
    #local var to track changed nodes, maps node ID to list of colors removed from that 
    #node's domain so we can replace them later if we backtrack
    revised = False
    
    # iterate domain of arc[0]
    for x in domains[arc[0]]:
        satisfied = False
        # iterate domain of arc[1]
        for y in domains[arc[1]]:
            if x != y:
                satisfied = True
                
        if not satisfied:
            domains[arc[0]].remove(x)
            altered[arc[0]].append(x)
            revised = True
            
        
    return (revised, altered)


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
                   
#Start of Graph Coloring Methods

def plot_graph():
    print(coords)
    xval = []
    yval = []
    
    for pt, edges in graph.items():
        for pt2 in edges:
            xval = [coords[pt][0], coords[pt2][0]]
            yval = [coords[pt][1], coords[pt2][1]]
            py.plot(xval, yval,'k:')
    py.title('Graph Coloring')
    py.xlabel("X-Axis")
    py.ylabel("Y-Axis")
    for i in range(len(coords)):
        xval[:]=[]
        yval[:]=[] 
        xval.append(coords[i][0])
        yval.append(coords[i][1])
        py.plot(xval, yval, marker='o',color=VertexColoringOtherAlgo(i), markersize = 10) 
    py.show()
    
def VertexColoringOtherAlgo(key):
    if colorNode[key][0] == 0:
        return 'r'
    elif colorNode[key][0] == 1:
        return 'b'
    elif colorNode[key][0] == 2:
        return 'g'
    else:
        return 'y'
        
def plot_graph_minConflict():
    print(coords)
    xval = []
    yval = []
    
    for pt, edges in graph.items():
        for pt2 in edges:
            xval = [coords[pt][0], coords[pt2][0]]
            yval = [coords[pt][1], coords[pt2][1]]
            py.plot(xval, yval,'k:')
    py.title('Graph Coloring')
    py.xlabel("X-Axis")
    py.ylabel("Y-Axis")
    for i in range(len(coords)):
        xval[:]=[]
        yval[:]=[] 
        xval.append(coords[i][0])
        yval.append(coords[i][1])
        py.plot(xval, yval, marker='o',color=VertexColoring(i), markersize = 10) 
    py.show()
    
    
def VertexColoring(v):
    if color[v] == 0 :
        return 'r'
    elif color[v] == 1 :
        return 'b'
    elif color[v] == 2 :
        return 'g'
    else : 
        return 'y'
        
#End of Graph Display Methods    
    
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
    
    
def min_conflict_unit_test():
    tests_passed = True
    global OP_COUNT
    # Successful coloring test
    for node1, v in graph.items():
        for node2 in v:
            if color[node1] == color[node2]:
                print("Adjacent nodes have same color!")
                print("Node " + str(node1) + ": " + str(color[node1]) + ", Node " + str(node2)) + ": " + str(color[node2])
                tests_passed = False
                
    if tests_passed:
        print("Unit tests successfully passed!")
        #print("Total Operations: ")
        #print(OP_COUNT)
    else:
        print("Unit tests failed!")
        #print("Total Operations: ")
        #print(OP_COUNT)
        
    OP_COUNT = 0
    
    
def ga_unit_test(solution):
    tests_passed = True
    global OP_COUNT
    # Successful coloring test
    for node1, v in graph.items():
        for node2 in v:
            if solution[node1] == solution[node2]:
                print("Adjacent nodes have same color!")
                print("Node " + str(node1) + ": " + str(color[node1]) + ", Node " + str(node2)) + ": " + str(color[node2])
                tests_passed = False
                
    if tests_passed:
        print("Unit tests successfully passed!")
        print("Number of Generations: ")
        print(OP_COUNT)
    else:
        print("Unit tests failed!")
        #print("Total Operations: ")
        #print(OP_COUNT)
        
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
        
        
def run_experiment_min_conflicts(num_colors):
    for i in range(1, 2):
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
        
        # Run Min Conflicts
        print("##############################################################")
        print("Running Min Conflicts - " + str(num_colors) + " colors, "+ str(num_points) + " points")
        #BackTracking(4)
        print(get_time())
        min_conflicts(100, num_colors)
        print(get_time())
        
        min_conflict_unit_test()
        
        
def run_experiment_genetic_algorithm(num_colors):
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
        
        # Run GA
        print("##############################################################")
        print("Running GA - " + str(num_colors) + " colors, "+ str(num_points) + " points")
        #BackTracking(4)
        print(get_time())
        solution = genetic_algorithm(num_colors, 10000, 30)
        print(solution)
        print(get_time())
        
        if solution:
            ga_unit_test(solution)
            
            
def test_runs():
    generate_points(10)
    print("Coordinates:")
    print(coords.items())
    
    # Determine Euclidean Distances
    calculate_distances()
    
    # Connect Edges
    build_graph()
    print("Graph: ")
    print(graph)
    
    # Create Adjacency Matrix
    matrix_creation()
    
    # BEGIN ALGORITHMS
    # SIMPLE BACKTRACKING
    print("##############################################################")
    print("Running Simple Backtracking  - 4 colors, 10 points")
    print(get_time())
    print(modified_backtracking(4, "simple")) 
    print(get_time())
    unit_tests()
    
    print("##############################################################")
    print("Running Simple Backtracking  - 3 colors, 10 points")
    print(get_time())
    print(modified_backtracking(3, "simple")) 
    print(get_time())
    unit_tests()
    
    # BACKTRACKING W/ FORWARD CHECKING
    print("##############################################################")
    print("Running Backtracking w/ Forward Checking  - 4 colors, 10 points")
    print(get_time())
    print(modified_backtracking(4, "forward")) 
    print(get_time())
    unit_tests()
    
    print("##############################################################")
    print("Running Backtracking w/ Forward Checking  - 3 colors, 10 points")
    print(get_time())
    print(modified_backtracking(3, "forward")) 
    print(get_time())
    unit_tests()
    
    # BACKTRACKING W/ MAC
    print("##############################################################")
    print("Running Backtracking w/ MAC  - 4 colors, 10 points")
    print(get_time())
    print(modified_backtracking(4, "mac")) 
    print(get_time())
    unit_tests()
    
    print("##############################################################")
    print("Running Backtracking w/ MAC  - 3 colors, 10 points")
    print(get_time())
    print(modified_backtracking(3, "mac")) 
    print(get_time())
    unit_tests()
    
    # MIN CONFLICTS
    print("##############################################################")
    print("Running Min Conflicts - 4 colors, 10 points")
    print(get_time())
    min_conflicts(1000, 4)
    print(get_time())
    min_conflict_unit_test()
    
    print("##############################################################")
    print("Running Min Conflicts - 3 colors, 10 points")
    print(get_time())
    min_conflicts(1000, 3)
    print(get_time())
    min_conflict_unit_test()

    
    # GENETIC ALGORITHM
    print("##############################################################")
    print("Running Genetic Algorithm - 4 colors, 10 points")
    print("Population size - 30")
    print(get_time())
    solution = genetic_algorithm(4, 10000, 30)
    print(solution)
    print(get_time())
    if solution:
        ga_unit_test(solution)
        
    print("##############################################################")
    print("Running Genetic Algorithm - 3 colors, 10 points")
    print("Population size - 30")
    print(get_time())
    solution = genetic_algorithm(3, 10000, 30)
    print(solution)
    print(get_time())
    if solution:
        ga_unit_test(solution)
    
   
   
def main():
    #run_experiment_simple_backtracking(3)
    #run_experiment_simple_backtracking(4)
    #run_experiment_backtracking_forward_checking(3)
    #run_experiment_backtracking_forward_checking(4)
    #run_experiment_backtracking_MAC(3)
    #run_experiment_backtracking_MAC(4)
    #run_experiment_min_conflicts(3)
    #run_experiment_min_conflicts(4)
    #run_experiment_genetic_algorithm(3)
    #run_experiment_genetic_algorithm(4)
    test_runs()
    pass
    
if __name__ == '__main__':
    main()
