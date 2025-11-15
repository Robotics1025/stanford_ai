from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.
#   > Please read the docstrings for in `mapUtil.py`, especially for the CityMap class

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # The start state simply contains the start location with no additional memory needed
        return State(location=self.startLocation)

    def isEnd(self, state: State) -> bool:
        # Check if the current location has the desired endTag
        return self.endTag in self.cityMap.tags[state.location]

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        """
        Note we want to return a list of *3-tuples* of the form:
            (actionToReachSuccessor: str, successorState: State, cost: float)
        Our action space is the set of all named locations, where a named location 
        string represents a transition from the current location to that new location.
        """
        # Get all possible successors and their costs from the current location
        successors = []
        for nextLocation, cost in self.cityMap.distances[state.location].items():
            # Create a new state for each successor location
            successorState = State(location=nextLocation)
            # Action is the name of the location we're moving to
            successors.append((nextLocation, successorState, cost))
        return successors


########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. 

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Get a known location (first one in the map) for testing
    startLocation = next(iter(cityMap.tags.keys()))
    endTag = makeTag("amenity", "food")
    
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`. Note that tags 
    from the `startLocation` count towards covering the set of tags.

    Hint: naively, your `memory` representation could be a list of all locations visited.
    However, that would be too large of a state space to search over! Think 
    carefully about what `memory` should represent.
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # Get initial waypoint tags that are covered by the start location
        initial_covered = frozenset(
            tag for tag in self.waypointTags 
            if tag in self.cityMap.tags[self.startLocation]
        )
        return State(location=self.startLocation, memory=initial_covered)

    def isEnd(self, state: State) -> bool:
        # Check if we're at a location with endTag and have covered all waypoints
        return (self.endTag in self.cityMap.tags[state.location] and 
                len(state.memory) == len(self.waypointTags))

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        successors = []
        for nextLocation, cost in self.cityMap.distances[state.location].items():
            # Determine which waypoint tags are covered at the next location
            new_covered = set(state.memory)
            for tag in self.waypointTags:
                if tag in self.cityMap.tags[nextLocation]:
                    new_covered.add(tag)
            
            successorState = State(location=nextLocation, memory=frozenset(new_covered))
            successors.append((nextLocation, successorState, cost))
            
        return successors


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem with waypoints using the map of Stanford, 
    specifying your own `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 2b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # Plan a route from Gates Building to parking, passing by food and library
    gates_tag = makeTag("landmark", "gates")
    startLocation = locationFromTag(gates_tag, cityMap)
    if startLocation is None:
        raise ValueError("Could not find Gates building location")
    waypointTags = [makeTag("amenity", "food"), makeTag("amenity", "library")]
    endTag = makeTag("amenity", "parking")
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 4a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.
# See util.py for the class definitions and methods of Heuristic and SearchProblem.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            return problem.startState()

        def isEnd(self, state: State) -> bool:
            return problem.isEnd(state)

        def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # Get successors from original problem
            successors = []
            for action, nextState, cost in problem.actionSuccessorsAndCosts(state):
                # Add heuristic value of successor state to the cost
                h_cost = heuristic.evaluate(nextState)
                successors.append((action, nextState, cost + h_cost - heuristic.evaluate(state)))
            return successors

    return NewSearchProblem()


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute end locations
        self.endLocations = []
        for location in cityMap.tags:
            if endTag in cityMap.tags[location]:
                self.endLocations.append(location)

    def evaluate(self, state: State) -> float:
        if not self.endLocations:
            return 0
        # Find minimum straight-line distance to any end location
        minDistance = float('inf')
        for endLoc in self.endLocations:
            dist = computeDistance(self.cityMap.geoLocations[state.location], self.cityMap.geoLocations[endLoc])
            minDistance = min(minDistance, dist)
        return minDistance


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        # Solving this reversed shortest path problem will give us our heuristic,
        # as it estimates the minimal cost of reaching an end state from each state
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                return State(location="END")

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                return False

            def actionSuccessorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                successors = []
                if state.location == "END":
                    # Connect END state to all locations with endTag with cost 0
                    for location in cityMap.tags:
                        if endTag in cityMap.tags[location]:
                            successors.append((location, State(location=location), 0))
                else:
                    # Regular successor generation from current location
                    for nextLocation, cost in cityMap.distances[state.location].items():
                        successors.append((nextLocation, State(location=nextLocation), cost))
                return successors

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.

        reverseProblem = ReverseShortestPathProblem()
        ucs = UniformCostSearch()

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!

        ucs.solve(reverseProblem)  # Run UCS
        self.pastCosts = ucs.pastCosts

    def evaluate(self, state: State) -> float:
        return self.pastCosts.get(state, float('inf'))


if __name__ == "__main__":
    """
    Small CLI for running route examples from this module.

    Examples:
        python -m route.submission --demo shortest
        python -m route.submission --demo waypoints
        python -m route.submission --demo shortest --method astar
    """
    import argparse
    import json
    from mapUtil import getTotalCost, createStanfordMap

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["shortest", "waypoints"], default="shortest", help="which demo problem to run")
    parser.add_argument("--method", choices=["ucs", "astar"], default="ucs", help="which search method to run")
    parser.add_argument("--out", default="path.json", help="output path JSON for visualization")
    args = parser.parse_args()

    cityMap = createStanfordMap()

    if args.demo == "shortest":
        problem = getStanfordShortestPathProblem()
        waypointTags = []
    else:
        problem = getStanfordWaypointsShortestPathProblem()
        waypointTags = list(problem.waypointTags)

    # Choose algorithm
    if args.method == "ucs":
        searcher = UniformCostSearch()
        searchProblem = problem
    else:
        # reduce A* to UCS using StraightLineHeuristic if available
        heur = StraightLineHeuristic(problem.endTag, problem.cityMap)
        searchProblem = aStarReduction(problem, heur)
        searcher = UniformCostSearch()

    print(f"Running {args.method.upper()} on demo={args.demo} ...")
    searcher.solve(searchProblem)

    if searcher.pathCost is None:
        print("No path found.")
    else:
        start = problem.startLocation
        path = [start] + searcher.actions
        # Print path with tags
        doneWaypointTags = set()
        for loc in path:
            tagsStr = " ".join(problem.cityMap.tags.get(loc, []))
            for tag in problem.cityMap.tags.get(loc, []):
                if tag in waypointTags:
                    doneWaypointTags.add(tag)
            doneTagsStr = " ".join(sorted(doneWaypointTags))
            print(f"Location {loc} tags:[{tagsStr}]; done:[{doneTagsStr}]")

        total = getTotalCost(path, problem.cityMap)
        print(f"Total distance: {total}")

        # Save for visualization
        with open(args.out, "w") as f:
            json.dump({"waypointTags": waypointTags, "path": path}, f, indent=2)
        print(f"Wrote path to {args.out} (use visualization.py to view)")
