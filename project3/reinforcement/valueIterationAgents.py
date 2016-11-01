# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util, pdb

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
    	for _ in range(self.iterations):
            newIterationValues = util.Counter() # The new values
            for state in self.mdp.getStates():
                maxQValue = -float('inf')
                hasAction = len(self.mdp.getPossibleActions(state)) > 0
                for action in self.mdp.getPossibleActions(state):
                    QValue = self.computeQValueFromValues(state, action)
                    if QValue > maxQValue:
                        maxQValue = QValue
                if hasAction:
                    newIterationValues[state] = maxQValue
            self.values = newIterationValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        statesAndProbabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        qvalue = 0
        for nextState, p in statesAndProbabilities:
        	reward = self.mdp.getReward(state, action, nextState)
        	qvalue += p * (reward + self.discount * self.getValue(nextState))
        return qvalue



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
        	return None
        actions = self.mdp.getPossibleActions(state)
        bestAction = None
        bestValue = -float('inf')
        for action in actions:
        	tempValue = self.computeQValueFromValues(state, action)
        	if tempValue > bestValue:
        		bestValue = tempValue
        		bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for k in range(self.iterations):
			maxQValue = -float('inf')
			states = self.mdp.getStates()
			stateCount = len(states)
			stateIndex = k % stateCount
			if self.mdp.isTerminal(states[stateIndex]):
				continue

			for action in self.mdp.getPossibleActions(states[stateIndex]):
				QValue = self.computeQValueFromValues(states[stateIndex], action)
				if QValue > maxQValue:
					maxQValue = QValue
			self.values[states[stateIndex]] = maxQValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
    	# states = self.mdp.getStates()
    	# predecessors = {state: set() for state in states}
    	# for state in states:
    	# 	statesAndProb = self.mdp.getTransitionStatesAndProbs(state, self.theta)
    	# 	for s, p in statesAndProb:
    	# 		if p > 0:
    	# 			predecessors[s].add(state)

     #    priorityQ = util.PriorityQueue()
     #    for state in states:
     #    	if self.mdp.isTerminal(state):
     #    		continue
     #    	pdb.set_trace()
     #    	maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getAction(state)])
     #    	diff = abs(self.values[state] - maxQValue)
     #    	priorityQ.update((state, maxQValue), -diff) # pushing q value also to prevent re-computing

     #    for iteration in self.iterations:
     #    	if priorityQ.isEmpty():
     #    		return
     #    	(state, qValue) = priorityQ.pop()
     #    	if not self.mdp.isTerminal(state):
     #    		self.values[state] = qValue

     #    	for predecessor in predecessors[state]:
     #    		maxQValue = max([self.computeQValueFromValues(state, action) for action in self.mdp.getAction(state)])
     #    		diff = abs(self.values[state] - maxQValue)
     #    		if diff > self.theta:
     #    			priorityQ.update((state, maxQValue), -diff) # pushing q value also to prevent re-computing

        # Compute predecessors of all states
        predecessorsOf = dict()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    if probability > 0:
                        # if state can reach nextState, then state is a predecessor of nextState
                        if nextState in predecessorsOf:
                            predecessorsOf[nextState].add(state)
                        else:
                            predecessorsOf[nextState] = set([state])

        # Initialize PriorityQueue
        fringe = util.PriorityQueue()
        # For each non-terminal state do:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                bestAction = self.computeActionFromValues(state)
                diff = abs(self.values[state] - self.computeQValueFromValues(state, bestAction))
                # prioritize states that have a higher error, negate diff because using min heap
                # update() pushes if state doesn't exist, or updates its priority if it improved
                # push() is ok to use here since each state is different
                fringe.push(state, -diff)

        # For each iteration do:
        for _ in range(self.iterations):
            if fringe.isEmpty():
                return
            state = fringe.pop()
            # update state's value, if it is not a terminal state, in self.values
            if not self.mdp.isTerminal(state):
                bestAction = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, bestAction)
            # for each predecessor p of s, do:
            for p in predecessorsOf[state]:
                # a predecessor isn't a terminal state by definition, so no need to account for that
                bestAction = self.computeActionFromValues(p)
                diff = abs(self.values[p] - self.computeQValueFromValues(p, bestAction))
                if diff > self.theta:
                    # prioritize states that have a higher error, negate diff because using min heap
                    # update() pushes if state doesn't exist, or updates its priority if it improved
                    fringe.update(p, -diff)














