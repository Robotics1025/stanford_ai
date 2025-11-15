from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
  """
  A reflex agent chooses an action at each choice point by examining
  its alternatives via a state evaluation function.

  The code below is provided as a guide.  You are welcome to change
  it in any way you see fit, so long as you don't touch our method
  headers.
  """

  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(
        gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(
        len(scores)) if scores[index] == bestScore]
    # Pick randomly among the best
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current GameState (defined in pacman.py)
    and a proposed action and returns a rough estimate of the resulting successor
    GameState's value.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [
        ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
  """
  Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
    Returns the minimax action from the current gameState using self.depth
    and self.evaluationFunction.
    """
    def minimax(state: GameState, depth: int, agentIndex: int) -> tuple:
        # Check for terminal states
        if state.isWin() or state.isLose() or not state.getLegalActions(agentIndex):
            return state.getScore(), None
            
        # If we've hit our depth limit and it's pacman's turn
        if depth == 0 and agentIndex == 0:
            return self.evaluationFunction(state), None

        # Get legal actions for current agent
        actions = state.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        
        # Update depth if we've processed all agents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        # Initialize best score and action based on agent type
        if agentIndex == 0:  # Pacman - maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in actions:
                score, _ = minimax(state.generateSuccessor(agentIndex, action), 
                                 nextDepth, nextAgent)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
        else:  # Ghost - minimizing agent
            bestScore = float('inf')
            bestAction = None
            for action in actions:
                score, _ = minimax(state.generateSuccessor(agentIndex, action), 
                                 nextDepth, nextAgent)
                if score < bestScore:
                    bestScore = score
                    bestAction = action
                    
        return bestScore, bestAction

    # Start minimax from Pacman's perspective (agent 0)
    _, action = minimax(gameState, self.depth, 0)
    return action

######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
  Your minimax agent with alpha-beta pruning (problem 2)
  You may reference the pseudocode for Alpha-Beta pruning here:
  en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """

    def alphabeta(state: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> tuple:
        # Check for terminal states
        if state.isWin() or state.isLose() or not state.getLegalActions(agentIndex):
            return state.getScore(), None
            
        # If we've hit our depth limit and it's pacman's turn
        if depth == 0 and agentIndex == 0:
            return self.evaluationFunction(state), None

        # Get legal actions for current agent
        actions = state.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        
        # Update depth if we've processed all agents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        # Initialize best score and action based on agent type
        if agentIndex == 0:  # Pacman - maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in actions:
                score, _ = alphabeta(state.generateSuccessor(agentIndex, action), 
                                   nextDepth, nextAgent, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    break
            return bestScore, bestAction
        else:  # Ghost - minimizing agent
            bestScore = float('inf')
            bestAction = None
            for action in actions:
                score, _ = alphabeta(state.generateSuccessor(agentIndex, action), 
                                   nextDepth, nextAgent, alpha, beta)
                if score < bestScore:
                    bestScore = score
                    bestAction = action
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    break
            return bestScore, bestAction

    # Start alpha-beta from Pacman's perspective (agent 0)
    _, action = alphabeta(gameState, self.depth, 0, float('-inf'), float('inf'))
    return action

######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
  Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
    Returns the expectimax action using self.depth and self.evaluationFunction

    All ghosts should be modeled as choosing uniformly at random from their
    legal moves.
    """
    def expectimax(state: GameState, depth: int, agentIndex: int) -> tuple:
        # Check for terminal states
        if state.isWin() or state.isLose() or not state.getLegalActions(agentIndex):
            return state.getScore(), None
            
        # If we've hit our depth limit and it's pacman's turn
        if depth == 0 and agentIndex == 0:
            return self.evaluationFunction(state), None

        # Get legal actions for current agent
        actions = state.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        
        # Update depth if we've processed all agents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        if agentIndex == 0:  # Pacman - maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in actions:
                score, _ = expectimax(state.generateSuccessor(agentIndex, action), 
                                    nextDepth, nextAgent)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            return bestScore, bestAction
        else:  # Ghost - chance node
            total = 0
            for action in actions:
                score, _ = expectimax(state.generateSuccessor(agentIndex, action), 
                                    nextDepth, nextAgent)
                total += score
            avg = total / len(actions) if actions else 0
            return avg, None

    # Start expectimax from Pacman's perspective (agent 0)
    _, action = expectimax(gameState, self.depth, 0)
    return action

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
  Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """
  # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE


# Abbreviation
better = betterEvaluationFunction

'''
To play, type:   python pacman.py
'''