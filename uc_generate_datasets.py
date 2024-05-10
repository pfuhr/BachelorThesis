from helper import binomial_coefficient, labels, my_simulate_for_sbi
import CRN as crn
import helper as helper
import torch as torch
from sbi.inference import SNPE, prepare_for_sbi
from sbi import utils as utils
from sbi.neural_nets.embedding_nets import FCEmbedding

from functools import partial
import time
from math import sqrt
import numpy as np
import time
from sbi import analysis as analysis
from helper_ppc import PPC

#in principle works, other colours?

def uc():
    model_type = 1
    run = 3
    exp = 1
            
    for n in range(2, 10):
        num_parameters = 2*binomial_coefficient(n,2) + n
          
        my_model = crn.MyModel(n)
        eval_dataset = torch.load(f"datasets/datasets{n}/eval_dataset_{n}.pth")
    
        for cur_round in [1000, 10000, 100000]:   
            #Load density_estimator
            density_estimator = torch.load(f'models_exp1/models{n}/models{n}_{model_type}_{run}/density_estimator{n}_{model_type}_{run}_{cur_round}.pth')

            device = next(density_estimator.parameters()).device
            print(f"density estimator is on device: {device}")
                
            prior = utils.BoxUniform(low=0 * torch.ones(my_model.reactions_count), high = 1 * torch.ones(my_model.reactions_count), device = device)

            #currently used model architecture
            embedding_net = FCEmbedding(input_dim = my_model.species_count, num_layers=3, num_hiddens = 70) #we can also customize the embedding net
            neural_posterior = utils.posterior_nn( 
                model="nsf", embedding_net=embedding_net
            )
            inference = SNPE(prior=prior, density_estimator=neural_posterior, device = device)

            posterior = inference.build_posterior(density_estimator=density_estimator, sample_with = "direct") #direct gave the fastest results

                       
            for j in range(1000):            
                true_theta = eval_dataset['thetas'][j:(j+1)]
                print('shape of true theta', true_theta.shape)
                true_x = eval_dataset['xs'][j:(j+1)]                
                posterior.set_default_x(true_x)

                sampled_theta = posterior.sample((10000,)) #should be of shape (1000, dimension theta)
                print("shape of sampled theta:", sampled_theta.shape)
                    
                
                if j == 0:
                    sampled_thetas = sampled_theta.unsqueeze(0) # adds a leading dimension
                    true_thetas = true_theta 
                else:
                    sampled_thetas = torch.cat((sampled_thetas, sampled_theta.unsqueeze(0)), dim = 0)
                    true_thetas = torch.cat((true_thetas, true_theta), dim = 0)

            print(f"Final shape of sampled_thetas: {sampled_thetas.shape} \n Final shape of true_thetas: {true_thetas.shape}")    
            D = {
              'sampled_thetas' : sampled_thetas,
              'true_thetas': true_thetas
            }

            torch.save(D, f"uc_datasets/uc_datasets_exp{exp}/uc_datasets{n}/uc_datasets_{cur_round}.pth")


            #ppc_plots{cur_round}/ppc_plots{n}/ppc_plots{j}.png'




if __name__ == "__main__":
    begin = time.time()
    uc()
    end = time.time()
    print(f"Total Plotting took {end - begin} seconds.")
