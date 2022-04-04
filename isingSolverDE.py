
from audioop import cross
import random, sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class isingSolverDE():
    def __init__(self,NS=8,crossover_rate=0.7,N_epochs=100,NParallel=-1):
        self.NS = NS
        self.crossover_rate = crossover_rate
        self.N_epochs = N_epochs
        self.NParallel = NParallel
        
    def init_pop(self):
        S2 = np.round(np.random.random((self.D,self.NS))) # Num_classes x Num_kernels x Population size
        S = np.ones((self.D,self.NS)) # Num_classes x Num_kernels x Population size
        # S[:,:int(NS/2)] = S2[:,:int(NS/2)]
        S[:,1:] = S2[:,1:]
        return S


    def ising(self,S,W,b):
        def ising_calc(s,W,b):
            s = np.expand_dims(s,1)
            s_T = s.T
            cost = -0.5*np.sum((np.matmul(s,s_T)*W))-np.sum(b*s)
            return cost

        print('Calculating Ising values...')
        S_costs = Parallel(n_jobs=self.NParallel,backend="threading")(delayed(ising_calc)(S[:,s_indx],W,b) for s_indx in range(S.shape[1]))
        return np.expand_dims(np.asarray(S_costs),0)

    def evolution(self,S):
        NS = S.shape[1] # pop size
        D = S.shape[0] 
        F_thr = 0.9
        
        parents = np.asarray([np.random.permutation(np.arange(NS))[:3] for i in range(NS)])
        F = np.random.random((D,NS))<F_thr
        mask = np.logical_and(F,np.logical_or(S[:,parents[:,1]], S[:,parents[:,2]]))
        ev = (1-S[:,parents[:,0]])*mask + S[:,parents[:,0]]*np.logical_not(mask)

        cr = (np.random.rand(D,NS)<self.crossover_rate)
        mut_keep = ev*cr
        pop_keep = S*np.logical_not(cr)
        CS = mut_keep + pop_keep
        return CS

    def selection(self,S,CS,S_costs,CS_costs):
        best_indx = CS_costs<S_costs

        best_indx = np.where(best_indx[0]==True)[0]
        S[:,best_indx] = CS[:,best_indx]
        S_costs[0,best_indx] = CS_costs[0,best_indx]
        
        S_best_cost = np.min(S_costs)
        S_best_indx = np.argmin(S_costs)

        S_best = S[:,S_best_indx]
        S_avg_cost = np.mean(S_costs)
        return S, S_costs, S_best, S_best_cost, S_avg_cost


    def optimize(self,weights,bias):
        self.D = weights.shape[1] # Number of states/dimensionality of data
        assert self.D==bias.shape[1], "Dimensionality of weights and bias do not match."
        S = self.init_pop() # DxNS
        S_costs = self.ising(S,weights,bias) # 1,NS

        for epoch in range(self.N_epochs):
            CS = self.evolution(S) # candidate S
            CS_costs = self.ising(CS,weights,bias) # 1,NS
            S, S_costs, S_best, S_best_cost, S_avg_cost = self.selection(S,CS,S_costs,CS_costs) # select next states
            print('Epoch:',epoch)
            print('Avg. Cost is:',S_avg_cost)
            print('Best Cost is:',S_best_cost)
            print(10*'-')
        print('Best Solution is:',S_best)
        print('Best Solution Cost is:',S_best_cost)

        return S_best,S_best_cost






# def reader(path_weights,path_bias,D,mode,W,b):
#     if mode=='call':
#         W = W
#         b = b
#     elif mode=='random':
#         W = np.random.uniform(-1,1,size=[D,D])
#         W = W + np.transpose(W) - 2*np.diag(W.diagonal())
#         b = np.random.uniform(-1,1,size=[1,D])
#     elif mode=='read':
#         W=np.loadtxt('W.txt')
#         b=np.loadtxt('b.txt')
#     else:
#         raise('Wrong mode!')
#     if W.shape[0]!=D or b.shape[0]!=D:
#         ValueError('Weights/bias dimensions does not match D!')
#     return W,b

