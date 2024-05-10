import CRN as crn
import helper as helper
import torch as torch
from sbi.inference import SNPE
from sbi import utils as utils

import time

from codecarbon import EmissionsTracker
from codecarbon import track_emissions
from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding

n = 2
exp = 3
@track_emissions(output_file=f"emissions/emissions_model_{n}_consistent_dataset.csv")
def train_models():
    #fix index
    
    model_type = 5

    # load Data and embedding_net

    print("loading datasets ... \n")
    dataset = torch.load(f"datasets/datasets{n}/consistent_dataset_{n}.pth") 
    theta = dataset['thetas']
    x = dataset['xs']
    dim = torch.numel(x[0])

    #HYPERPARAMETERS

    stop_after_epochs = 5
    batch_size = 50 #not included yet
    device = 'cuda'

    #MODEL

    my_model = crn.MyModel(n)

    #PREPARE    
    prior = utils.BoxUniform(low=0 * torch.ones(my_model.reactions_count), high = 1 * torch.ones(my_model.reactions_count), device = device)
    #simulator, prior = prepare_for_sbi(simulator_function, prior)

    #SETUP THE INFERENCE ALGORITHM
    embedding_net = FCEmbedding(input_dim = dim, num_layers=3, num_hiddens = 70) #we can also customize the embedding net
    # instantiate the neural density estimator
    neural_posterior = utils.posterior_nn( #this can be further specified, e.g model = 'nsf'
        model="nsf", embedding_net=embedding_net
    )


    #Overview
    round1 = 1000
    round2 = 10000
    round3 = 100000
    rounds = [round1, round2, round3]
    
    
    for run in range(1):
        for cur_round in rounds: 
            print(f'gpu training for n={n}, model_type = {model_type}, run = {run}, round = {cur_round} \n')
                        
            #cur ROUND
            print(f"Round {cur_round}: \n defining inference workflow and specifying device")

            inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)

            #INFERENCE
            print("appending simulations ... \n")

            inference = inference.append_simulations(theta[:cur_round], x[:cur_round])

            print(f"training on {cur_round} samples ... \n")

            t1 = time.time()
            density_estimator = inference.train(stop_after_epochs=stop_after_epochs)
            t2 = time.time()

            print(f"... took {t2-t1} seconds.\n")

            torch.save(density_estimator, f'models_exp{exp}/models{n}/models{n}_{model_type}_{run}/density_estimator{n}_{model_type}_{run}_{cur_round}.pth')
            
if __name__ == "__main__":
    t1 = time.time()
    train_models()
    t2 = time.time()
    print(f"Everything took {t2-t1} seconds.")