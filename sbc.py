from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot
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


#in principle works, other colours?

def sbc():
    model_type = 1
    run = 3
    exp = 1
    ks_pvals = {}
    c2st_ranks = {}
    c2st_dap = {}    
    ranks_dict = {}

    for n in range(2,10):
        num_parameters = 2*binomial_coefficient(n,2) + n
          
        my_model = crn.MyModel(n)
        if exp == 1:
            eval_dataset = torch.load(f"datasets/datasets{n}/eval_dataset_{n}.pth")
        elif exp == 2:
            eval_dataset = torch.load(f"datasets/datasets{n}/cut_eval_dataset_{n}.pth")
        elif exp == 3:
            eval_dataset = torch.load(f"datasets/datasets{n}/consistent_eval_dataset_{n}.pth")
            
        for cur_round in [1000, 10000, 100000]:   
            #Load density_estimatorz
            density_estimator = torch.load(f'models_exp{exp}/models{n}/models{n}_{model_type}_{run}/density_estimator{n}_{model_type}_{run}_{cur_round}.pth')

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
            
            thetas = eval_dataset['thetas']
            xs = eval_dataset['xs']                
            
            thetas = thetas.to(device)
            xs = xs.to(device)

            num_posterior_samples = 1_0000
            ranks, dap_samples = run_sbc(
                thetas, xs, posterior, num_posterior_samples=num_posterior_samples
            )

            check_stats = check_sbc(
                ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
            )
                
            print(
                f"""kolmogorov-smirnov p-values \n
                check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"""
            )
            ks_pvals[(n, cur_round)] = check_stats['ks_pvals'].numpy()

            print(
                f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()}"
            )
            c2st_ranks[(n, cur_round)] = check_stats['c2st_ranks'].numpy()

            print(f"- c2st accuracies check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}")
            c2st_dap[(n, cur_round)] = check_stats['c2st_dap'].numpy()

            ranks_dict[(n, cur_round)] = ranks

            f, ax = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,  # by passing None we use a heuristic for the number of bins.
            )

            f.savefig(f'plots/plots_exp{exp}/sbc_plots/sbc_plots{n}/sbc_plot{cur_round}_2.png')
            
    torch.save(ks_pvals, f'metric_results_exp{exp}/sbc/ks_pvals_2.pth')
    torch.save(c2st_ranks, f'metric_results_exp{exp}/sbc/c2st_ranks_2.pth')
    torch.save(c2st_dap, f'metric_results_exp{exp}/sbc/c2st_dap_2.pth')
    torch.save(ranks_dict, f'metric_results_exp{exp}/sbc/ranks_dict_2.pth')

if __name__ == "__main__":
    begin = time.time()
    sbc()
    end = time.time()
    print(f"Total Computation took {end - begin} seconds.")
