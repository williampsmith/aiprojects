# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        farthestFood = closestFood = 1
        farthestCapsule = closestCapsule = 1
        newFoodCount = 1
        remainingFood = successorGameState.getNumFood()

        if remainingFood:
          newFoodCount = remainingFood

        newScore = successorGameState.getScore()
        remainingCapsules = successorGameState.getCapsules()

        maxDistance = newFood.height + newFood.width
        offensiveGhostDist = [maxDistance]
        scaredGhostDist =[maxDistance]

        for ghost in newGhostStates:
          ghostPos = ghost.getPosition()
          distanceToGhost = abs(ghostPos[0]-newPos[0])+abs(ghostPos[1]-newPos[1])

          if ghost.scaredTimer < distanceToGhost:
            if distanceToGhost < 2:
              distanceToGhost = -500
            offensiveGhostDist.append(distanceToGhost)
          else:
            scaredGhostDist.append(distanceToGhost + 1)
        closestOffensiveGhost = min(offensiveGhostDist)
        closestScaredGhost = min(scaredGhostDist)
        foodDist = [abs(food[0] - newPos[0]) + abs(food[1] - newPos[1]) for food in newFood.asList()]

        if foodDist:
          farthestFood = max(foodDist)
          closestFood = min(foodDist)
        capsuleDist = [abs(capsule[0] - newPos[0]) + abs(capsule[1] - newPos[1]) for capsule in remainingCapsules]

        if capsuleDist:
          closestCapsule = min(capsuleDist)

        return 20*newScore + 0.1*closestOffensiveGhost + 10*(1.0/newFoodCount) + 20*(1.0/closestFood) + 10*(1.0/farthestFood) + 10*(1.0/closestCapsule) + 20*(1.0/closestScaredGhost)

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.calcValue(gameState, 0, 0)[1]

    def calcValue(self, state, agent, depth):
      isMaximizingAgent = False
      if agent == 0:
        isMaximizingAgent = True
      elif agent == state.getNumAgents():
        isMaximizingAgent = True
        agent = 0
        depth += 1
      if state.isWin() or state.isLose() or depth == self.depth:
        return (self.evaluationFunction(state), None)
      if isMaximizingAgent:
        return self.max(state, agent, depth)
      else:
        return self.min(state, agent, depth)

    def max(self, state, agent, depth):
      bestValueAndAction = (-float('inf'), None)

      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth)[0], action)
        bestValueAndAction = max(bestValueAndAction, nextValueAndAction, key=lambda x: x[0])
      return bestValueAndAction

    def min(self, state, agent, depth):
      bestValueAndAction = (float('inf'), None)

      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth)[0], action)
        bestValueAndAction = min(bestValueAndAction, nextValueAndAction, key=lambda x: x[0])

      return bestValueAndAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.calcValue(gameState, 0, 0, -float('inf'), float('inf'))[1]

    def calcValue(self, state, agent, depth, a, b):
      isMaximizingAgent = False
      if agent == 0:
        isMaximizingAgent = True
      elif agent == state.getNumAgents():
        isMaximizingAgent = True
        agent = 0
        depth += 1
      if state.isWin() or state.isLose() or depth == self.depth:
        return (self.evaluationFunction(state), None)
      if isMaximizingAgent:
        return self.max(state, agent, depth, a, b)
      else:
        return self.min(state, agent, depth, a, b)

    def max(self, state, agent, depth, a, b):
      bestValueAndAction = (-float('inf'), None)

      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth, a, b)[0], action)
        bestValueAndAction = max(bestValueAndAction, nextValueAndAction, key=lambda x: x[0])
        if bestValueAndAction[0] > b:
          return bestValueAndAction
        a = max(a, bestValueAndAction[0])
      return bestValueAndAction

    def min(self, state, agent, depth, a, b):
      bestValueAndAction = (float('inf'), None)

      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth, a, b)[0], action)
        bestValueAndAction = min(bestValueAndAction, nextValueAndAction, key=lambda x: x[0])
        if bestValueAndAction[0] < a:
          return bestValueAndAction
        b = min(b, bestValueAndAction[0])

      return bestValueAndAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.calcValue(gameState, 0, 0)[1]

    def calcValue(self, state, agent, depth):
      isMaximizingAgent = False
      if agent == 0:
        isMaximizingAgent = True
      elif agent == state.getNumAgents():
        isMaximizingAgent = True
        agent = 0
        depth += 1
      if state.isWin() or state.isLose() or depth == self.depth:
        return (self.evaluationFunction(state), None)
      if isMaximizingAgent:
        return self.max(state, agent, depth)
      else:
        return self.exp(state, agent, depth)

    def max(self, state, agent, depth):
      bestValueAndAction = (-float('inf'), None)

      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth)[0], action)
        bestValueAndAction = max(bestValueAndAction, nextValueAndAction, key=lambda x: x[0])
      return bestValueAndAction

    def exp(self, state, agent, depth):
      expectation = 0

      p = 1.0 / len(state.getLegalActions(agent)) # probability 
      for action in state.getLegalActions(agent):
        nextState = state.generateSuccessor(agent, action)
        nextValueAndAction = (self.calcValue(nextState, agent+1, depth)[0], action) 
        expectation = p * nextValueAndAction[0]
      return (expectation, None)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

