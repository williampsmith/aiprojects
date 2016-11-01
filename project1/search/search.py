# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from pdb import *

class SearchNode:
    def __init__(self, state):
        self.state = state
        self.path = list() # a list of states, from start to current node
        self.directions = list() # a list of sequential directions
        self.backward_cost = 0 # the total cost from start to current node

    def expand(self, problem):
        return problem.getSuccessors(self.state)

    def addToPath(self, this_path):
        self.path.append(this_path)



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# general fringe push function to cover all kinds of search
def insert(node, fringe):
    if hasattr(fringe, 'priorityFunction'): # if running A*
        fringe.push(node)
    elif hasattr(fringe, 'heap'): # if running UCS
        fringe.push(node, node.backward_cost)
    else:
        fringe.push(node)

def generalSearch(problem, fringe):
    closed = set() # a set of states
    initialState = problem.getStartState()
    start_node = SearchNode(initialState)
    start_node.addToPath(initialState) # path to the start node is itself
    insert(start_node, fringe)

    while True:
        #set_trace()
        if fringe.isEmpty():
            return "FAIL!" # fixme later
        node = fringe.pop()
        if problem.isGoalState(node.state):
            return node.directions
        #set_trace()
        if node.state not in closed:
            closed.add(node.state)
            for successor in node.expand(problem):
                #set_trace()
                # successor form: (new_state, action, step_cost)
                child = SearchNode(successor[0])
                child.path = node.path + [successor[0]]
                child.directions = node.directions + [successor[1]]
                child.backward_cost = node.backward_cost + successor[2]
                insert(child, fringe)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # for testing
    return generalSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return generalSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return generalSearch(problem, util.PriorityQueue())

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def priorityFunction(node):
        return heuristic(node.state, problem) + node.backward_cost
    return generalSearch(problem, util.PriorityQueueWithFunction(priorityFunction))



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
