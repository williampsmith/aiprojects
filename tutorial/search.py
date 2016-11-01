import pdb

graph = {'A': ['B', 'C'],
         'B': ['C', 'E'],
         'C': ['D'],
         'D': ['F'],
         'E': ['F'],
         'F': ['C']}

# need to implement closing the nodes and failure returning/checking

def bfs(start, end):
	fringe = [[start]]
	path = [start]
	while path[-1] != end:
		fringe = expand(fringe) # expand the appropriate fringe node
		path = fringe[0]
	return path

def expand(fringe):
	expanded_path = fringe[0]
	fringe = fringe[1:] + [(expanded_path + [n]) for n in graph[expanded_path[-1]]]
	#pdb.set_trace()
	return fringe
