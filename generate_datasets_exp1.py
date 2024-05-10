import CRN as crn
import helper as helper
import torch as torch
from sbi.inference import prepare_for_sbi
from sbi import utils as utils
from helper import my_simulate_for_sbi

#set index: 
n=2


n_simulations = 100000 #Number of simulations
#n_evaluations = 10000

print(f'Generating simulation datasets for index {n} as in generate_datasets_2.py.')

#MODEL
my_model = crn.MyModel(n)
#print(my_model)
my_simulation = crn.MySimulation( my_model, algorithm="mnrm", max_t=0.1, number_of_observations=200)
#print(my_simulation)

#PREPARE    
simulator_function = my_simulation.simulate 
prior = utils.BoxUniform(low=0 * torch.ones(my_simulation.input_dim), high = 1 * torch.ones(my_simulation.input_dim))
simulator, prior = prepare_for_sbi(simulator_function, prior)

#simulations
theta, x = my_simulate_for_sbi(prior, n_simulations, simulator)
#eval_theta, eval_x = theta[(n_simulations - n_evaluations):], x[(n_simulations - n_evaluations):]

#this line was missing before
#theta, x = theta[:(n_simulations - n_evaluations)], x[:(n_simulations - n_evaluations)]

dataset = {
    'thetas': theta,
    'xs': x
}

#eval_dataset = {
#    'thetas': eval_theta,
#    'xs': eval_x
#}

torch.save(dataset, f'results/results{n}/hd_results/dataset_{n}.pth')
#torch.save(eval_dataset, f'results/results{n}/hd_results/eval_dataset_{n}.pth')

