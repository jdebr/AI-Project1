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


# Represents an undirected graph.  Key is node ID, value is a list of           
# node IDs that share an edge.
# {nodeID: [nodeIDs...]}
graph = {}


# Maps node ID to some color value
# {nodeID: 'color'}
color = {}


# Maps node ID to a tuple representing Cartesian coordinates
# {nodeID: (x, y)}
coords = {}


# Maps node ID to a list of tuples which represent NodeID and distance 
# {nodeID: [(nodeID, distance)...]}  
distance = defaultdict(list)


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
        
        
def main():
    generate_points(5)
    calculate_distances()
    
    for key, value in coords.items():
        print(key, value)
        
    print(distance)
    
    
if __name__ == '__main__':
    main()