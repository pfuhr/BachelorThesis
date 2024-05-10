import math
import numpy as np
from sbi.utils.sbiutils import seed_all_backends
import torch as torch
from joblib import Parallel, delayed
import multiprocessing
import time
from torch import Tensor


# Function to calculate binomial coefficient (n choose k)
def binomial_coefficient(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
#i ranges from 1 to n-1 and j from 0 to i-1
#0 <=j < i <= n-1
def get_species_index(j, i, n):
    if j > i :
        print(f"WARNING: j = {j} > i = {i} in function get_species_index. \n We will return now get_species_index({j},{i}).")
        return get_species_index(i, j, n)
    elif j == i :
        print(f"ERROR: j = {j} = i = {i} in function get_species_index. We will return {i}.")
        return i
    else:
        index = n-1
        #0 <=l < k < n
        for k in range(1,n):
            for l in range(k):
                index += 1
                if k == i and l == j:
                    return index
    print(f"No matching condition found in function get_species_index for j={j}, i={i}, n={n}")
    return None

#get reaction index in pre respectively post arrays    
               
def get_reaction_index(j, i, n, forward: bool): #forward is a bool indicating wether it is a forward or backward reaction
    if j > i :
        print(f"WARNING: j = {j} > i = {i} in function get_reaction_index. \n We will return now get_reaction_index({j},{i}).")
        return get_reaction_index(i, j, n, forward)
    elif j == i :
        print(f"ERROR: j = {j} = i = {i} in function get_reaction_index. We will return {i}.")
        return i
    else:
        index = n-1
        #0 <=l < k < n
        for k in range(1,n):
            for l in range(k):
                index += 1
                if k == i and l == j:
                    if forward: 
                        return index
                    else:
                        return index + 1
                #passed both forward and backward reactions R_ij , R_ij'
                index += 1
    print(f"No matching condition found in function get_reaction_index for j={j}, i={i}, n={n}, forward={forward}")
    return None

# compute possible reaction combinations scaled by c (prospensity function), corresponds to de facto reaction rates
def prospensity(x, pre, c):
    #we need to ensure that x >= pre for each possible reaction
    reaction_possible = np.all((pre <= x), axis=1)
    reactant_combinations = (x**pre).prod(1) * c
    return np.where(reaction_possible, reactant_combinations, 0.0) # computes number of total possible reactant combinations * kinetic rate c

#gillespie SSA Algorithm to simulate Biochem. Reaction Network
def gillespie(x, c, pre, post, max_t):
    """
    Gillespie simulation

    Parameters
    ----------

    x: 1D array of size n_species
        The initial numbers.

    c: 1D array of size n_reactions
        The reaction rates.

    # a reaction can be understood as a a reactant vector of size n_species (pre) a product vector of n_species and a reaction rate c

    pre: array of size n_reactions x n_species 
        What is to be consumed.

    post: array of size n_reactions x n_species
        What is to be produced

    max_t: int
        Timulate up to time max_t

    Returns
    -------
    t, X: 1d array, 2d array
        t: The time points.
        X: The history of the species.
           ``X.shape == (t.size, x.size)``

    """
    t = 0
    t_store = [t]
    x_store = [x.copy()]
    S = post - pre #stochiometric vector

    count_reactions = 0

    

    while t < max_t:
        h_vec = prospensity(x, pre, c) #computes prospensity function for each reaction
        
        h0 = h_vec.sum() 
        if h0 == 0: # no reactant combination 
            break
        
        #print("h0", h0)
        delta_t = np.random.exponential(1.0/h0) 
        # no reaction can occur any more
        if not np.isfinite(delta_t):
            t_store.append(max_t)
            x_store.append(x)
            break
        reaction = np.random.choice(c.size, p=h_vec / h0)
        
        t = t + delta_t
        x = x + S[reaction]
        count_reactions +=1
        if np.any(x<0):
            print("negative concentrations, x:", x)

        t_store.append(t)
        x_store.append(x)

    return np.array(t_store), np.array(x_store), count_reactions


def mnrm(x, c, pre, post, max_t):
    #1 set t=0. For each k set Tk =0
    count_reactions = 0
    t = 0
    t_store = [t]
    x_store = [x.copy()]
    n_reactions = c.size #e.g
    S = post - pre #stochiometric vector
    
    T_vec = np.zeros(n_reactions)
    
    #2 Calculate the propensity function, ak, for each reaction.

    a_vec = prospensity(x, pre, c)

    #4 For each k, set Pk ~ exp(1)
    P_vec = np.random.exponential(scale=1.0, size=n_reactions)

    while (t < max_t):
        #5 For each k, set d(tk)=(Pk-Tk)/ak.
        
        delta_t_vec = (P_vec - T_vec)/a_vec
  
        #6 Set d*=mink {d(tk)} and let d(tmü) be the time where the
        #minimum is realized.
        delta_t_mü = np.min(delta_t_vec)
        mü = np.argmin(delta_t_vec)

        #7 Set t=t+d* and update the number of each molecular
        #species according to reaction mü.
        t += delta_t_mü
        count_reactions +=1
        x = x + S[mü]
        if np.any(x<0):
            print("negative concentrations, x:", x)

        t_store.append(t)
        x_store.append(x)
        #8 For each k, set Tk=Tk+ak*d*.
        T_vec = T_vec + delta_t_mü*a_vec
        #9 For reaction mü,  and set P=P+ x ~exp(1).
        e = np.random.exponential(scale=1.0)
        P_vec[mü] += e
        #10 Recalculate the propensity functions, ak.
        a_vec = prospensity(x, pre, c)
                
    return np.array(t_store), np.array(x_store), count_reactions

def my_simulate_for_sbi(prior, n_simulations, simulator_function)->Tensor:
    theta = prior.sample((n_simulations,))
    theta = theta.cpu()
    #print('theta: \n', theta)
    
    batches = torch.split(theta, 1, dim=0)
    #print('batches: \n', batches)
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    print(f"We are simulating across {num_cores} CPU cores.")

    def simulator_seeded(theta:Tensor, seed)->Tensor:
        seed_all_backends(seed)
        return simulator_function(theta)

    batch_seeds = torch.randint(high=1_000_000, size=(len(batches),))

    t1=time.time()

    try:
        simulation_outputs = Parallel(n_jobs=-1)(
            delayed(simulator_seeded)(batch, batch_seed)
            for batch, batch_seed in zip(batches, batch_seeds)
        )
            
    finally:
        t2 = time.time()
        print(f'It took {t2 - t1} seconds to run {n_simulations} simulations. ')
    #print(simulation_outputs)
    x = torch.cat(simulation_outputs, dim=0)

    return theta, x

def my_simulate_for_sbi2(theta, simulator_function)->Tensor:
    #theta = prior.sample((n_simulations,))
    theta = theta.cpu()
    #print('theta: \n', theta)
    
    batches = torch.split(theta, 1, dim=0)
    #print('batches: \n', batches)
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    print(f"We are simulating across {num_cores} CPU cores.")

    def simulator_seeded(theta:Tensor, seed)->Tensor:
        seed_all_backends(seed)
        return simulator_function(theta)

    batch_seeds = torch.randint(high=1_000_000, size=(len(batches),))

    t1=time.time()

    try:
        simulation_outputs = Parallel(n_jobs=-1)(
            delayed(simulator_seeded)(batch, batch_seed)
            for batch, batch_seed in zip(batches, batch_seeds)
        )
            
    finally:
        t2 = time.time()
        print(f'It took {t2 - t1} seconds to run {n_simulations} simulations. ')
    #print(simulation_outputs)
    x = torch.cat(simulation_outputs, dim=0)

    return theta, x


def labels(n): #returns labels for plots

    ds = [f'd_{i}' for i in range(1,n+1)]
    ks = []
    for i in range(2, n+1): 
        for j in range(1, i):
            ks += [f'k_{j}{i}', f'k_{j}{i}^-1']     
    return ds + ks  


