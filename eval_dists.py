#Evaluation file to compute distance of estimated versus true sample and normalized log probabilities true samples
#run 1-5 of model 1
#f'models_exp1/models{n}/models{n}_{model_type}_{run+1}/density_estimator{n}_{model_type}_{run+1}_{round3}.pth')

#File to compute the expected value (theta* - theta), where theta* ~ prior and theta ~ q(theta| x*), where x* is the output of the simulation simulated on theta 
#It evaluates models trained on the gpu relatively fast < 5 mins
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
import time
from Eval import eval_mean, eval_median, batch_eval_mean, batch_eval_median, expected_log_prob, compute_converged_metric
import time
from codecarbon import track_emissions


@track_emissions(output_file=f"emissions/evaluation_dists_model_type_3.csv")
def evaluate():
    round1 = 1000
    round2 = 10000
    round3 = 100000
    rounds = [round1, round2, round3]
    model_type = 3
    run = 0

    mean_dists1 = np.array([])
    mean_dists2 = np.array([])
    mean_dists3 = np.array([])

    mean_dists_unscaled1 = np.array([])
    mean_dists_unscaled2 = np.array([])
    mean_dists_unscaled3 = np.array([])

    med_dists1 = np.array([])
    med_dists2 = np.array([])
    med_dists3 = np.array([])

    med_dists_unscaled1 = np.array([])
    med_dists_unscaled2 = np.array([])
    med_dists_unscaled3 = np.array([])

    epsilon = 0.01
    iterations = 10
    step = 1000 # must be bigger than 0
    for n in range(2,10):
        expected_log_probs_n = np.array([])
        all_expected_log_probs = np.array([])
        print(f"evaluating index {n}")
        my_model = crn.MyModel(n)
        scale = sqrt(my_model.reactions_count)

        eval_dataset = torch.load(f"datasets/datasets{n}/cut_eval_dataset_{n}.pth")
        eval_thetas = eval_dataset['thetas']
        eval_xs = eval_dataset['xs']
        
        for cur_round in rounds:
            mean_dists_across_runs = np.array([])
            med_dists_across_runs = np.array([])
            for run in range(5):
    
                #ROUND1
                print(f"dist evaluation for n = {n}, cur_round = {cur_round}, run = {run} ")

                #Load density_estimator
                density_estimator = torch.load(f'models_exp2/models{n}/models{n}_{model_type}_{run}/density_estimator{n}_{model_type}_{run}_{cur_round}.pth')
                
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

                #loading evaluation dataset and moving to device
                eval_thetas = eval_thetas.to(device)
                eval_xs = eval_xs.to(device)

                #We are checking convergence by checking x_i - x_(i-1) < epsilon 
                scale = sqrt(my_model.reactions_count)

                mean_dist = partial(batch_eval_mean, posterior = posterior, n_posterior_samples = 1000, scale = scale)
                med_dist = partial(batch_eval_median, posterior = posterior, n_posterior_samples = 1000, scale = scale)
                t1=time.time()
                result_mean_dist = compute_converged_metric(mean_dist, eval_thetas, eval_xs, step, epsilon, iterations).item()
                result_med_dist = compute_converged_metric(med_dist, eval_thetas, eval_xs, step, epsilon, iterations).item()
                t2 = time.time()
                print(f"For run = {run}, n = {n}, in round = {cur_round} : \n scaled mean distance: {result_mean_dist}. \n scaled median distance: {result_med_dist}. \n We took {t2-t1} seconds to converge. ")
                

                mean_dists_across_runs = np.append(mean_dists_across_runs, result_mean_dist)
                med_dists_across_runs = np.append(med_dists_across_runs, result_med_dist)

            
            if cur_round == 1000:
                mean_dists1 = np.append(mean_dists1, np.mean(mean_dists_across_runs))
                med_dists1 = np.append(med_dists1, np.mean(med_dists_across_runs))
                mean_dists_unscaled1 =  np.append(mean_dists_unscaled1, np.mean(mean_dists_across_runs)*scale)
                med_dists_unscaled1 = np.append(med_dists_unscaled1, np.mean(med_dists_across_runs)*scale)
                
            if cur_round == 10000:
                mean_dists2 = np.append(mean_dists2, np.mean(mean_dists_across_runs))
                med_dists2 = np.append(med_dists2, np.mean(med_dists_across_runs))
                mean_dists_unscaled2 =  np.append(mean_dists_unscaled2, np.mean(mean_dists_across_runs)*scale)
                med_dists_unscaled2 = np.append(med_dists_unscaled2, np.mean(med_dists_across_runs)*scale)
            
            if cur_round == 100000:
                mean_dists3 = np.append(mean_dists3, np.mean(mean_dists_across_runs))
                med_dists3 = np.append(med_dists3, np.mean(med_dists_across_runs))
                mean_dists_unscaled3 =  np.append(mean_dists_unscaled3, np.mean(mean_dists_across_runs)*scale)
                med_dists_unscaled3 = np.append(med_dists_unscaled3, np.mean(med_dists_across_runs)*scale)
            
        



    np.savez(f'metric_results_exp2/dists_model_type_{model_type}/mean_dists_theta.npz', m1000 = mean_dists1, m10000 = mean_dists2, m100000 = mean_dists3)
    np.savez(f'metric_results_exp2/dists_model_type_{model_type}/mean_dists_theta_unscaled.npz', m1000 = mean_dists_unscaled1, m10000 = mean_dists_unscaled2, m100000 = mean_dists_unscaled3)

    np.savez(f'metric_results_exp2/dists_model_type_{model_type}/med_dists_theta.npz', m1000 = med_dists1, m10000 = med_dists2, m100000 = med_dists3)
    np.savez(f'metric_results_exp2/dists_model_type_{model_type}/med_dists_theta_unscaled.npz', m1000 = med_dists_unscaled1, m10000 = med_dists_unscaled2, m100000 = med_dists_unscaled3)

if __name__ == "__main__":
    begin = time.time()
    evaluate()
    end = time.time()
    print(f"Total evaluation took {end - begin} seconds.")

