import copy
from util import CSP, get_or_variable, CourseBulletin, Profile
from typing import Dict, List

############################################################
# Problem 1

# Hint: Take a look at the CSP class and the CSP examples in util.py for
# reference; they should be helpful.
# Hint: The following general examples should be helpful references.
# Add a variable to the CSP:
#   csp = CSP()
#   csp.add_variable(VAR_NAME, VAR_DOMAIN)
# Add a unary factor:
#   csp.add_unary_factor(VAR_NAME, factor_function)
# Add a binary factor:
#   csp.add_binary_factor(VAR1_NAME, VAR2_NAME, factor_function)
#
# Notice that the third input to add_binary_factor is a function!
# factor_function should return 0 if the constraints are unsatisfied, or
# 1 or weight (for problem 3) otherwise.
# You can define factor_function with lambdas or a nested function in Python.
# The following are example functions corresponding to binary factors:
# Using lambdas:
#   csp.add_binary_factor(VAR1_NAME, VAR2_NAME, lambda x, y: x == y)
# Using nested functions:
#   def are_equal(x,y):
#     return x == y
#   csp.add_binary_factor(VAR1_NAME, VAR2_NAME, are_equal)
# See util.py for more examples.


def create_chain_csp(n: int) -> CSP:
    # name variables as x1, x2, ..., xn
    variables = [f'x{i}' for i in range(1, n+1)]
    csp = CSP()
    # Problem 1c
    # Create variables x1..xn with domain {0,1}
    for v in variables:
        csp.add_variable(v, [0, 1])
    # Add XOR (inequality) binary factors between consecutive variables
    for i in range(len(variables) - 1):
        csp.add_binary_factor(variables[i], variables[i+1], lambda a, b: a != b)
    # END_YOUR_CODE
    return csp


############################################################
# Problem 2

def create_nqueens_csp(n: int = 8) -> CSP:
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_factor().

    Hint: When defining the factor function (the last argument to
          csp.add_binary_factor()), you can use any variable that
          exists during the moment in which you create the factor
          function. For example, you can define a lambda function
          such as
            lambda a, b: (a + x) > (b + y)
          as long as x and y are defined at the instant in which
          you create this lambda function.

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver.
    """
    csp = CSP()
    # Problem 2a
    # One variable per column (0..n-1), domain is row indices 0..n-1
    for col in range(n):
        csp.add_variable(col, list(range(n)))

    # Binary constraints: different rows and not on same diagonal
    for i in range(n):
        for j in range(i + 1, n):
            dist = j - i
            def factor(a, b, dist=dist):
                return (a != b) and (abs(a - b) != dist)
            csp.add_binary_factor(i, j, factor)
    # END_YOUR_CODE
    return csp

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)


class BacktrackingSearch:
    def reset_results(self) -> None:
        """
        This function resets the statistics of the different aspects of the
        CSP solver. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []
        self.allOptimalAssignments = []

    def print_stats(self) -> None:
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print(f'Found {self.numOptimalAssignments} optimal assignments \
                    with weight {self.optimalWeight} in {self.numOperations} operations')
            print(
                f'First assignment took {self.firstAssignmentNumOperations} operations')
        else:
            print(
                "No consistent assignment to the CSP was found. The CSP is not solvable.")

    def get_delta_weight(self, assignment: Dict, var, val) -> float:
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert var not in assignment
        w = 1.0
        if self.csp.unaryFactors[var]:
            w *= self.csp.unaryFactors[var][val]
            if w == 0:
                return w
        for var2, factor in list(self.csp.binaryFactors[var].items()):
            if var2 not in assignment:
                continue  # Not assigned yet
            w *= factor[val][assignment[var2]]
            if w == 0:
                return w
        return w

    def satisfies_constraints(self, assignment: Dict, var, val) -> bool:
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return whether or not assigning the variable with the proposed new value
        still statisfies all of the constraints.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        return self.get_delta_weight(assignment, var, val) != 0

    def solve(self, csp: CSP, mcv: bool = False, ac3: bool = False) -> None:
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param ac3: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.ac3 = ac3

        # Reset solutions from previous search.
        self.reset_results()

        # The dictionary of domains of every variable in the CSP.
        self.domains = {
            var: list(self.csp.values[var]) for var in self.csp.variables}

        # Perform backtracking search.
        self.backtrack({}, 0, 1)
        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment: Dict, numAssigned: int, weight: float) -> None:
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in self.csp.variables:
                newAssignment[var] = assignment[var]
            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                    self.allOptimalAssignments.append(newAssignment)
                else:
                    self.numOptimalAssignments = 1
                    self.allOptimalAssignments = [newAssignment]
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)
        # Get an ordering of the values.
        ordered_values = self.domains[var]

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.ac3:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned +
                                   1, weight * deltaWeight)
                    del assignment[var]
        else:
            # Arc consistency check is enabled. This is helpful to speed up 3c.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    # create a deep copy of domains as we are going to look
                    # ahead and change domain values
                    localCopy = copy.deepcopy(self.domains)
                    # fix value for the selected variable so that hopefully we
                    # can eliminate values for other variables
                    self.domains[var] = [val]

                    # enforce arc consistency
                    self.apply_arc_consistency(var)

                    self.backtrack(assignment, numAssigned +
                                   1, weight * deltaWeight)
                    # restore the previous domains
                    self.domains = localCopy
                    del assignment[var]

    def get_unassigned_variable(self, assignment: Dict):
        """
        Given a partial assignment, return a currently unassigned variable.

        @param assignment: A dictionary of current assignment. This is the same as
            what you've seen so far.

        @return var: a currently unassigned variable. The type of the variable
            depends on what was added with csp.add_variable
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in self.csp.variables:
                if var not in assignment:
                    return var
        else:
            # Problem 2b
            # Heuristic: most constrained variable (MCV)
            best_var = None
            best_count = None
            # iterate in the CSP variable order to break ties by lowest index
            for var in self.csp.variables:
                if var in assignment:
                    continue
                # count the number of values that would satisfy constraints
                cnt = 0
                for val in self.domains[var]:
                    if self.satisfies_constraints(assignment, var, val):
                        cnt += 1
                # initialize or update best
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    best_var = var
            return best_var

    def apply_arc_consistency(self, var) -> None:
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.
        """

        def remove_inconsistent_values(var1, var2):
            removed = False
            # the binary factor must exist because we add var1 from var2's neighbor
            factor = self.csp.binaryFactors[var1][var2]
            for val1 in list(self.domains[var1]):
                # Note: in our implementation, it's actually unnecessary to check unary factors,
                #       because in get_delta_weight() unary factors are always checked.
                if (self.csp.unaryFactors[var1] and self.csp.unaryFactors[var1][val1] == 0) or \
                        all(factor[val1][val2] == 0 for val2 in self.domains[var2]):
                    self.domains[var1].remove(val1)
                    removed = True
            return removed

        queue = [var]
        while len(queue) > 0:
            curr = queue.pop(0)
            for neighbor in self.csp.get_neighbor_vars(curr):
                if remove_inconsistent_values(neighbor, curr):
                    queue.append(neighbor)


def create_sum_variable(csp: CSP, name: str, variables: List, maxSum: int) -> tuple:
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain range(0, maxSum+1), such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed. You
        can use it to get the auxiliary variables' domain

    @return result: The name of a newly created variable with domain range
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """

    result = ('sum', name, 'aggregated')
    csp.add_variable(result, list(range(maxSum + 1)))

    if len(variables) == 0:
        csp.add_unary_factor(result, lambda x: x == 0)
        return result

    domain = []
    for i in range(maxSum + 1):
        for j in range(i, maxSum + 1):
            domain.append((i, j))

    for i in range(len(variables)):
        csp.add_variable(('sum', name, str(i)), domain)

    csp.add_unary_factor(('sum', name, '0'), lambda x: x[0] == 0)

    for i in range(len(variables)):
        f = ('sum', name, str(i))
        csp.add_binary_factor(f, variables[i], lambda x, y: x[1] == x[0] + y)

    for i in range(len(variables) - 1):
        f0 = ('sum', name, str(i))
        f1 = ('sum', name, str(i + 1))
        csp.add_binary_factor(f0, f1, lambda x, y: x[1] == y[0])

    csp.add_binary_factor(
        ('sum', name, str(len(variables) - 1)), result, lambda x, y: x[1] == y)

    return result

############################################################
# Problem 3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.


class SchedulingCSPConstructor:
    def __init__(self, bulletin: CourseBulletin, profile: Profile):
        """
        Saves the necessary data.

        @param bulletin: Stanford Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """

        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp: CSP) -> None:
        """
        Adding the variables into the CSP. Each variable, (request, quarter),
        can take on the value of one of the courses requested in request or None.
        For instance, for quarter='Aut2013', and a request object, request, generated
        from 'CS221 or CS246', (request, quarter) should have the domain values
        ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
        then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
        we are not taking either of them in 'Aut2013'.

        @param csp: The CSP where the additional constraints will be added to.
        """

        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_variable((request, quarter), request.cids + [None])

    def add_bulletin_constraints(self, csp: CSP) -> None:
        """
        Add the constraints that a course can only be taken if it's offered in
        that quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """

        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter),
                                     lambda cid: cid is None or
                                     self.bulletin.courses[cid].is_offered_in(quarter))

    def add_norepeating_constraints(self, csp: CSP) -> None:
        """
        No course can be repeated. Coupling with our problem's constraint that
        only one of a group of requested course can be taken, this implies that
        every request can only be satisfied in at most one quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """

        for request in self.profile.requests:
            for quarter1 in self.profile.quarters:
                for quarter2 in self.profile.quarters:
                    if quarter1 == quarter2:
                        continue
                    csp.add_binary_factor((request, quarter1), (request, quarter2),
                                          lambda cid1, cid2: cid1 is None or cid2 is None)

    def get_basic_csp(self) -> CSP:
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one quarter.

        @return csp: A CSP where basic variables and constraints are added.
        """

        csp = CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        self.add_norepeating_constraints(csp)
        return csp

    def add_quarter_constraints(self, csp: CSP) -> None:
        """
        If the profile explicitly wants a request to be satisfied in some given
        quarters, e.g. Aut2013, then add constraints to not allow that request to
        be satisfied in any other quarter. If a request doesn't specify the
        quarter(s), do nothing.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3a
        # If a request specifies allowed quarters, ensure that any variable
        # (request, quarter) can only take a non-None value if quarter is in
        # request.quarters. If request.quarters is empty, do nothing.
        for request in self.profile.requests:
            if not request.quarters:
                continue
            allowed = set(request.quarters)
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter), lambda cid, q=quarter, allowed=allowed: cid is None or q in allowed)
        # END_YOUR_CODE

    def add_request_weights(self, csp: CSP) -> None:
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """

        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter),
                                     lambda cid: request.weight if cid != None else 1.0)

    def add_prereq_constraints(self, csp: CSP) -> None:
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that *all courses in req.prereqs are
        being requested*. Note that if our parser inferred that one of your
        requested course has additional prerequisites that are also being
        requested, these courses will be added to req.prereqs. You will be notified
        with a message when this happens. Also note that req.prereqs apply to every
        single course in req.cids. If a course C has prerequisite A that is requested
        together with another course B (i.e. a request of 'A or B'), then taking B does
        not count as satisfying the prerequisite of C. You cannot take a course
        in a quarter unless all of its prerequisites have been taken *before* that
        quarter. You should take advantage of get_or_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """

        # Iterate over all request courses
        for req in self.profile.requests:
            if len(req.prereqs) == 0:
                continue
            # Iterate over all possible quarters
            for quarter_i, quarter in enumerate(self.profile.quarters):
                # Iterate over all prerequisites of this request
                for pre_cid in req.prereqs:
                    # Find the request with this prerequisite
                    for pre_req in self.profile.requests:
                        if pre_cid not in pre_req.cids:
                            continue
                        # Make sure this prerequisite is taken before the requested course(s)
                        prereq_vars = [(pre_req, q)
                                       for i, q in enumerate(self.profile.quarters) if i < quarter_i]
                        v = (req, quarter)
                        orVar = get_or_variable(
                            csp, (v, pre_cid), prereq_vars, pre_cid)
                        # Note this constraint is enforced only when the course is taken
                        # in `quarter` (that's why we test `not val`)
                        csp.add_binary_factor(
                            orVar, v, lambda o, val: not val or o)

    def add_unit_constraints(self, csp: CSP) -> None:
        """
        Add constraints to the CSP to ensure that course units are correctly assigned.
        This means meeting two conditions:

        1- If a course is taken in a given quarter, it should be taken for
           a number of units that is within bulletin.courses[cid].minUnits/maxUnits.
           If not taken, it should be 0.
        2- In each quarter, the total number of units of courses taken should be between
           profile.minUnits/maxUnits, inclusively.
           You should take advantage of create_sum_variable() to implement this.

        For every requested course, you must create a variable named (courseId, quarter)
        (e.g. ('CS221', 'Aut2013')) and its assigned value is the number of units.
        This variable is how our solution extractor will obtain the number of units,
        so be sure to add it.

        For a request 'A or B', if you choose to take A, then you must use a unit
        number that's within the range of A.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3b
        # For each requested course and quarter, add a variable (cid, quarter)
        # with domain [0..maxUnits] where 0 means not taken in that quarter.
        # Also add binary factors between (request, quarter) and (cid, quarter)
        # to enforce that if request variable says cid then units in range
        # otherwise units == 0.

        # First create all (cid, quarter) variables
        for req in self.profile.requests:
            for cid in req.cids:
                for quarter in self.profile.quarters:
                    course = self.bulletin.courses[cid]
                    domain = [0] + list(range(course.minUnits, course.maxUnits + 1))
                    csp.add_variable((cid, quarter), domain)

        # Add binary factors between request vars and (cid, quarter)
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                req_var = (req, quarter)
                for cid in req.cids:
                    course = self.bulletin.courses[cid]
                    def make_factor(cid_local, course_local):
                        def factor(chosen_cid, units):
                            # If request not choosing this course, units must be 0
                            if chosen_cid != cid_local:
                                return units == 0
                            # If chosen, units must be within course min/max
                            return units >= course_local.minUnits and units <= course_local.maxUnits
                        return factor
                    csp.add_binary_factor(req_var, (cid, quarter), make_factor(cid, course))

        # Now for each quarter, create a sum variable aggregating all (cid, quarter)
        # and constrain it to be within profile.minUnits and profile.maxUnits
        for quarter in self.profile.quarters:
            vars_in_quarter = []
            for req in self.profile.requests:
                for cid in req.cids:
                    vars_in_quarter.append((cid, quarter))
            if len(vars_in_quarter) == 0:
                continue
            # compute an upper bound for max sum to keep create_sum_variable efficient
            max_possible = sum(self.bulletin.courses[cid].maxUnits for cid, _ in vars_in_quarter)
            sum_var = create_sum_variable(csp, ('quarter', quarter), vars_in_quarter, max_possible)
            # enforce min and max inclusive
            csp.add_unary_factor(sum_var, lambda s, mn=self.profile.minUnits, mx=self.profile.maxUnits: s >= mn and s <= mx)
        # END_YOUR_CODE

    def add_all_additional_constraints(self, csp: CSP) -> None:
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_quarter_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_unit_constraints(csp)

''' commands used:
    1.python run_p2.py
    2.python run_p2.py --nqueens
    3.python test_nqueens.py
    4.python run_p3.py profile3b.txt
'''