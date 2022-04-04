import numpy as np
from isingSolverDE import isingSolverDE

'''
## Arguments:
# NS = 8                         # Population size, must be >= 4 and even number
# crossover_rate = 0.7            # Cross-over rate [0,1]; Larger more diversity
# N_epochs = 100                    # Max number of generations (N_epochs)
# D = 50                     # Number of states (d)
# NParallel = -1            # for parallel computing
# W is numpy array of weights   DxD
# b is numpy array of bias 1xD

## Note:
User can read the W and b from file or pass the values 
'''


#--- Generating random weights and bias values -----#
D = 10
W = np.random.uniform(-1,1,size=[D,D])
W = W + np.transpose(W) - 2*np.diag(W.diagonal())
b = np.random.uniform(-1,1,size=[1,D])
#---------------------------------------------------#


#--- Calling the Model ---#
solver = isingSolverDE(NS=8, crossover_rate=0.7, N_epochs=100, NParallel=-1)

#--- Calling the Optimizer ---#
State_best, State_best_cost = solver.optimize(weights=W,bias=b)

