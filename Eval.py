#File to compute the expected value (theta* - theta), where theta* ~ prior and theta ~ q(theta| x*), where x* is the output of the simulation simulated on theta 
#It evaluates models trained on the gpu relatively fast < 5 mins
import CRN as crn
import helper as helper
#from Val_Eval import sbc, calc_mses_meddists, calc_mses_meddists_prior, evaluate
import torch as torch
#from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
#from sbi import utils as utils
#from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding
#from sbi.analysis import check_sbc, run_sbc #, get_nltp, sbc_rank_plot
#from helper import my_simulate_for_sbi
from functools import partial
#from joblib import Parallel, delayed
import time
from math import sqrt
#prior, _, __ = process_prior(prior)


# computing the following metrics: E[theta* - theta_0], (theta_0, x_0) ~ p(theta, x), theta* ~ p(theta | x_0)
#n_posterior_samples = 1000


def eval_mean(theta, x, posterior, n_posterior_samples): #for theta, x one-dimensional
    
    posterior_samples = posterior.sample((n_posterior_samples,), x=x, show_progress_bars=False)
    expanded_theta = theta.expand_as(posterior_samples)

    # Compute Euclidean distance along the last dimension
    distances = torch.sqrt(torch.sum((expanded_theta - posterior_samples)**2, dim=-1))

    # Calculate mean distance across the batch
    mean_distance = torch.tensor([torch.mean(distances)])

    return mean_distance

def eval_median(theta, x, posterior, n_posterior_samples): #for theta, x one-dimensional
    posterior_samples = posterior.sample((n_posterior_samples,), x=x, show_progress_bars=False)
    expanded_theta = theta.expand_as(posterior_samples)

    # Compute Euclidean distance along the last dimension
    distances = torch.sqrt(torch.sum((expanded_theta - posterior_samples)**2, dim=-1))

    # Calculate mean distance across the batch
    median_distance = torch.tensor([torch.median(distances)])

    return median_distance

def batch_eval_mean(theta, x, posterior, n_posterior_samples, scale = 1):

    my_eval = partial(eval_mean, posterior = posterior, n_posterior_samples = n_posterior_samples)
    
    thetas = torch.split(theta, 1, dim=0)
        
    xs = torch.split(x, 1, dim=0)
    
    #evaluation_outputs = Parallel(n_jobs=-1)(
    #    delayed(my_eval)(theta_0, x_0)
    #    for theta_0, x_0 in zip(thetas, xs)
    #)

    evaluation_outputs = [my_eval(theta_0, x_0) for theta_0, x_0 in zip(thetas, xs)]
    
    evaluation_outputs = torch.cat(evaluation_outputs, dim=0)
    mean_distance = torch.mean(evaluation_outputs)
    
    return mean_distance/scale

def batch_eval_median(theta, x, posterior, n_posterior_samples, scale):

    my_eval = partial(eval_median, posterior=posterior, n_posterior_samples=n_posterior_samples)
    
    thetas = torch.split(theta, 1, dim=0)
        
    xs = torch.split(x, 1, dim=0)
    
    evaluation_outputs = [my_eval(theta_0, x_0) for theta_0, x_0 in zip(thetas, xs)]
    
    evaluation_outputs = torch.cat(evaluation_outputs, dim=0)
    mean_median_distance = torch.mean(evaluation_outputs)
    
    return mean_median_distance/scale

def expected_log_prob(thetas, xs, posterior):
    log_probs = []
    for i in range(thetas.shape[0]):
        log_prob = posterior.log_prob(thetas[i], xs[i])
        log_probs.append(log_prob)

    log_probs = torch.tensor(log_probs)
    
    #log_probs = torch.tensor([posterior.log_prob(thetas[i], xs[i]) for i in range(thetas.shape[0])])
    return - torch.mean(log_probs)

def compute_converged_metric(metric, eval_thetas, eval_xs, step, epsilon, iterations):
    """
    metric should be a function that can be applied on a batch of thetas and xs
    iterations*step should be smaller than 10000
    prior, density_estimator, and the posterior_estimator object need to have the same device parameter
    """
    #We are checking convergence by checking x_i - x_(i-1) < epsilon 
    
    #first round:
    total = 0
    cur_result = 0
    converged = False

    for i in range(0, iterations):
        print(f"evaluating the {i+1}-th round of {step} true samples")
        cur_theta, cur_x = eval_thetas[(i*step):((i+1)*step)], eval_xs[(i*step):((i+1)*step)]
        
        new_result = (total * cur_result + step * metric(cur_theta, cur_x)) / (total+step)

        cur_epsilon = abs((new_result - cur_result))
        if cur_epsilon < epsilon:
            converged = True
            print(f"The evaluation metric converged after evaluating {(i+1)*step} true samples.")
            cur_result = new_result
            break

        total = total + step
        cur_result = new_result 

    if converged == False:
        print(f"With choice of epsilon = {epsilon}, we didn't converge after {iterations*step} true samples. The last epsilon value was {cur_epsilon}. \n")    

    return cur_result
