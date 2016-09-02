'''
CSCI 446 
Fall 2016
Project 1

@author: Joe DeBruycker
@author: Shriyansh Kothari
@author: Sara Ounissi
'''

import random

class Vertex(object):
    def __init__(self, x, y, color = None):
        self.x = x
        self.y = y
        self.color = color
        
    def set_color(self, color):
        self.color = color 
        
    def to_string(self):
        print("Position (x,y): (" + str(self.x) + ", " + str(self.y) + ")")
        print("Color: " + str(self.color))
        
def generate_graph(n):
    '''
    Generate a random undirected graph with n vertices, 
    return a dictionary of vertices representing graph
    '''
    graph = {}
    
    for i in range(n):
        randX = random.random()
        randY = random.random()
        v = Vertex(randX, randY)
        graph[i] = v 
        
    return graph
        
def main():
    g = generate_graph(5)
    
    for key, value in g.items():
        value.to_string()
    
if __name__ == '__main__':
    main()