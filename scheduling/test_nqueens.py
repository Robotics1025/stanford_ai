import submission

csp = submission.create_nqueens_csp(8)
alg = submission.BacktrackingSearch()
alg.solve(csp, mcv=False, ac3=False)
print('nqueens 8, ops without MCV/AC3:', alg.numOperations, 'solutions:', alg.numAssignments)
alg2 = submission.BacktrackingSearch()
alg2.solve(csp, mcv=True, ac3=False)
print('nqueens 8, ops with MCV:', alg2.numOperations, 'solutions:', alg2.numAssignments)
alg3 = submission.BacktrackingSearch()
alg3.solve(csp, mcv=True, ac3=True)
print('nqueens 8, ops with MCV+AC3:', alg3.numOperations, 'solutions:', alg3.numAssignments)
