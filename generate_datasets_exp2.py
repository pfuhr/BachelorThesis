import CRN as crn
import helper as helper
from Val_Eval import sbc, calc_mses_meddists, calc_mses_meddists_prior, evaluate
import torch as torch
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding
from sbi.analysis import check_sbc, run_sbc #, get_nltp, sbc_rank_plot
from helper import my_simulate_for_sbi
import time

#set index: 
for n in range(2,10):

    #MODEL

    my_model = crn.MyModel(n)


    shape = (200,my_model.species_count)

    def reshape(x):
        return torch.flatten((x.view(shape)[:,:n]))

    dataset = torch.load(f"datasets/datasets{n}/dataset_{n}.pth") 
    thetas = dataset['thetas']
    xs = dataset['xs']
    
    new_xs = torch.stack([reshape(xs[i]) for i in range(xs.shape[0])])
    cut_dataset = {
        'thetas': thetas,
        'xs': new_xs, 
    }

    torch.save(cut_dataset, f"datasets/datasets{n}/cut_dataset_{n}.pth")
    

    eval_dataset = torch.load(f"datasets/datasets{n}/eval_dataset_{n}.pth") 
    thetas = eval_dataset['thetas']
    xs = eval_dataset['xs'] 
    
    new_xs = torch.stack([reshape(xs[i]) for i in range(xs.shape[0])])
    cut_eval_dataset = {
        'thetas': thetas,
        'xs': new_xs, 
    }

    torch.save(cut_eval_dataset, f"datasets/datasets{n}/cut_eval_dataset_{n}.pth")
    

    

