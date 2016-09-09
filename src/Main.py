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


# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}
graph = defaultdict(list)


# Maps node ID to some color value
# {nodeID: 'color'}
color = {}


# Maps node ID to a tuple representing Cartesian coordinates
# {nodeID: (x, y)}
coords = {}


# Maps node ID to a list of tuples which represent NodeID and distance 
# {nodeID: [(nodeID, distance)...]}  
distance = defaultdict(list)


# List of existing line segments (edges)
lines = []


def generate_points(n):
    ''' Generates n sets of points randomly scattered on the 
    unit square and stores them as tuples mapped to integer IDs
    '''
    for i in range(n):
        randX = random.randint(1, 10)
        randY = random.randint(1, 10)
        v = (randX, randY)
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
        
        
def draw_graph():
    ''' Add edges to graph by selecting a random point and adding 
    an edge to the nearest point that doesn't already have an edge
    and such that it will not cross any other edge'''
    # List of node IDs to check for possible edges
    available_nodes = list(coords.keys())
    
    #print(available_nodes)
    while available_nodes:
        selected_point = random.choice(available_nodes)
        lineAdded = False
        
        while not lineAdded:
            if not distance[selected_point] :
                available_nodes.remove(selected_point)
                break
            else : 
                nearest_point = distance[selected_point][0][0]
                distance[selected_point].pop(0)
                
                # if no edge already exists
                if nearest_point not in graph[selected_point]:
                    # set coordinates for calculation
                    pt1 = coords[selected_point]
                    pt2 = coords[nearest_point]
                    x1 = pt1[0]
                    y1 = pt1[1]
                    x2 = pt2[0]
                    y2 = pt2[1]
                    
                    # Check if vertical line
                    if x1 == x2:
                        print("Vertical line!")
                        pass
                    # Check if any intersections
                    else:
                        A1 = y2 - y1
                        B1 = x1 - x2
                        C1 = A1 * x1 + B1 * y1
                        
                        # No lines on graph so no conflict
                        if not lines:
                            lines.append((A1, B1, C1, x1, x2, y1, y2))
                        # Check all other lines for intersection
                        else:
                            for line in lines:
                                A2 = line[0]
                                B2 = line[1]
                                C2 = line[2]
                                x3 = line[3]
                                x4 = line[4]
                                y3 = line[5]
                                y4 = line[6]
                                
                                det = (A1 * B2) - (A2 * B1)
                                if det == 0:
                                    #lines are parallel
                                    print("Parallel!")
                                    pass
                                else:
                                    x = (B2 * C1 - B1 * C2) / det
                                    y = (A1 * C2 - A2 * C1) / det
                                    
                                    # Check if on segment 1
                                    if x >= min(x1, x2) and x <= max(x1, x2):
                                        if y >= min(y1, y2) and y <= max(y1, y2):
                                            # Check if on segment 2
                                            if x >= min(x3, x4) and x <= max(x3, x4):
                                                if y >= min(y3, y4) and y <= max(y3, y4):
                                                    # this will intersect so move on
                                                    print("Intersection!")
                                                    break
                                    # No intersection so add edge and line
                                    graph[selected_point].append(nearest_point)
                                    lines.append((A1, B1, C1, x1, x2, y1, y2))
                                    lineAdded = True
                    
            
            
    
    
    
    
    
    #print(selected_point)
    #print(nearest_point)
        
def main():
    generate_points(10)
    calculate_distances()
    draw_graph()
    
    print(lines)
    
    for key, value in coords.items():
        print(key, value)
        
    #print(distance)
    
    #py.plot([4,5,6])
    #py.show()
    
    
if __name__ == '__main__':
    main()