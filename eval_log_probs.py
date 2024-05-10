#Evaluation file to compute normalized log probabilities of true samples for model without preembedding outputs

import CRN as crn
import helper as helper
import torch as torch
from sbi.inference import SNPE 
from sbi import utils as utils
from sbi.neural_nets.embedding_nets import FCEmbedding

from functools import partial
import time
from math import sqrt
import numpy as np
from Eval import eval_mean, eval_median, batch_eval_mean, batch_eval_median, expected_log_prob, compute_converged_metric
import time
from codecarbon import track_emissions

@track_emissions(output_file=f"emissions/evaluation_log_probs_model_type_1.csv")
def evaluate():
    print('log prob evaluation of model type 1 \n')
    
    round1 = 1000
    round2 = 10000
    round3 = 100000
    model_type = 1
    rounds = [round1, round2, round3]

    
    epsilon = 0.1
    iterations = 10
    step = 1000 # must be bigger than 0
    print("Are we even evaluating the right file? This is eval_process.py")
    #we are computing one array for each n, containing the expected average log probs for each round
    for n in range(2,10):
        begin_n = time.time()
        expected_log_probs = np.array([])
        y = True
        #all_expected_log_probs = np.array([])
        for cur_round in rounds:
            log_probs_across_runs = np.array([])
            for run in range(5):
                #print(f"evaluating index {n}")
                my_model = crn.MyModel(n)
                scale = sqrt(my_model.reactions_count)

                eval_dataset = torch.load(f"datasets/datasets{n}/eval_dataset_{n}.pth")
                eval_thetas = eval_dataset['thetas']
                eval_xs = eval_dataset['xs']
                
                #Load density_estimator
                density_estimator = torch.load(f'models_exp1/models{n}/models{n}_{model_type}_{run}/density_estimator{n}_{model_type}_{run}_{cur_round}.pth')

                device = next(density_estimator.parameters()).device
                #print(f"density estimator is on device: {device}")

                #my_simulation = crn.MySimulation( my_model, algorithm="mnrm", max_t=0.1, number_of_observations=200)
                    
                prior = utils.BoxUniform(low=0 * torch.ones(my_model.reactions_count), high = 1 * torch.ones(my_model.reactions_count), device = device)

                #currently used model architecture
                embedding_net = FCEmbedding(input_dim = my_model.species_count, num_layers=3, num_hiddens = 70) #we can also customize the embedding net
                neural_posterior = utils.posterior_nn( 
                    model="nsf", embedding_net=embedding_net
                )
                inference = SNPE(prior=prior, density_estimator=neural_posterior, device = device)

                posterior = inference.build_posterior(density_estimator=density_estimator, sample_with = "direct") #direct gave the fastest results

                #loading evaluation dataset and moving to device
                eval_thetas = eval_thetas.to(device)
                eval_xs = eval_xs.to(device)

                #We are checking convergence by checking x_i - x_(i-1) < epsilon 
                scale = sqrt(my_model.reactions_count)

                log_prob_posterior = partial(expected_log_prob, posterior = posterior)
                
                t1 = time.time()
                result_log_prob = compute_converged_metric(log_prob_posterior, eval_thetas, eval_xs, step, epsilon, iterations).item()
                print("The result_log_prob is", result_log_prob, " so we can confirm that the problem is or is not here")
                t2 = time.time()
                print(f"evaluating the log probs for n = {n}, run = {run}, round = {cur_round} took {t2-t1} seconds.")
                log_probs_across_runs = np.append(log_probs_across_runs, result_log_prob)
            expected_log_probs = np.append(expected_log_probs, np.mean(log_probs_across_runs))
            log_probs_across_runs = np.expand_dims(log_probs_across_runs, axis=0)
    
            if y == True:
                all_expected_log_probs = log_probs_across_runs
                y = False
            else:
                all_expected_log_probs = np.concatenate([all_expected_log_probs, log_probs_across_runs], axis = 0)

        print('expected_log_probs:', expected_log_probs, '\n')
        print('all_expected_log_probs', all_expected_log_probs)
        #log_probs_{n}.npz['expected_log_probs'] = np.array[avg_log_prob[1000], avg_log_prob[10000], avg_log_prob[100000]]

        np.savez(f'metric_results_exp1/log_probs_model_type_1/log_probs_{n}.npz', all_expected_log_probs=all_expected_log_probs, expected_log_probs=expected_log_probs)
        end_n = time.time()
        print(f"Completing evaluation of index {n} took {end_n - begin_n} seconds!")


if __name__ == "__main__":
    begin = time.time()
    evaluate()
    end = time.time()
    print(f"Total evaluation took {end - begin} seconds.")
