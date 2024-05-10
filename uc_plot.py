import time
from math import sqrt
import numpy as np
import time
import helper_uc as uch
import torch as torch
#in principle works, other colours?

def uc():
    model_type = 1
    run = 3
    exp = 1
    
    for n in range(2, 10):
        for cur_round in [1000, 10000, 100000]:   
            #Load density_estimator
            D = torch.load(f"uc_datasets/uc_datasets_exp{exp}/uc_datasets{n}/uc_datasets_{cur_round}.pth")        
            sampled_thetas = D['sampled_thetas']
            true_thetas = D['true_thetas']

            print(f"sampled_thetas.shape: {sampled_thetas.shape}")
            print(f"true_thetas.shape: {true_thetas.shape}")
            
            uc = uch.UC(sampled_thetas, true_thetas)
            values = uc.compare(0.1, 0.9, 'single')
            plotter = uch.Plot_with_labels(n)
            fig = plotter.plot(values)
            fig.savefig(f'plots/plots_exp{exp}/uc_plots/uc_plots{n}/plot_{cur_round}.png')


def uc2():
    for exp in [2]:
        value_dict80 = {}
        std_dict80 = {}
        value_dict95 = {}
        std_dict95 = {}
        value_dict99 = {}
        std_dict99 = {}

        for n in range(2, 10):
            
            for cur_round in [1000, 10000, 100000]:   
                
                D = torch.load(f"uc_datasets/uc_datasets_exp{exp}/uc_datasets{n}/uc_datasets_{cur_round}.pth")        
                sampled_thetas = D['sampled_thetas'].cpu()
                true_thetas = D['true_thetas'].cpu()

                print(f"sampled_thetas.shape: {sampled_thetas.shape}")
                print(f"true_thetas.shape: {true_thetas.shape}")
                
                uc = uch.UC(true_thetas = true_thetas, sampled_thetas=sampled_thetas)
                
                
                #80
                _ = uc.quantiles( q_values = torch.tensor([0.1, 0.9])) # , 0.025, 0.005, 0.975, 0.995])) 
                #width_80 = uc.my_quantiles[1] - uc.my_quantiles[0]
                #avg_width_80 = torch.mean(width_80, dim=0)
                values, stddevs = uc.compare()
                value_dict80[(n, cur_round)] = values #list with percentages of true theta in 80% confidence interval of estimated posterior
                std_dict80[(n, cur_round)] = stddevs

                #95
                _ = uc.quantiles( q_values = torch.tensor([0.025, 0.975]))
                #width_95 = uc.my_quantiles[1] - uc.my_quantiles[0]
                #avg_width_95 = torch.mean(width_95, dim=0)
                values, stddevs = uc.compare()
                value_dict95[(n, cur_round)] = values #list with percentages of true theta in 95% confidence interval of estimated posterior
                std_dict95[(n, cur_round)] = stddevs
                
                #99
                _ = uc.quantiles( q_values = torch.tensor([0.005, 0.995]))
                #width_99 = uc.my_quantiles[1] - uc.my_quantiles[0]
                #avg_width_99 = torch.mean(width_99, dim=0) #should be of shape theta_dim
                values, stddevs = uc.compare()
                value_dict99[(n, cur_round)] = values #list with percentages of true theta in 99% confidence interval of estimated posterior
                std_dict99[(n, cur_round)] = stddevs
                
                """
                avg_widths = {
                  'avg_width_80': avg_width_80, 
                  'avg_width_95': avg_width_95,
                  'avg_width_99': avg_width_99
                }
                #print(avg_widths)
                torch.save(avg_widths, f"metric_results_exp{exp}/uc_data/uc_data{n}/avg_widths{cur_round}.pth")
                """

                #if cur_round == 1000:
                #    arrays = values.unsqueeze(0)
                #else:
                #    arrays = torch.cat((arrays, values.unsqueeze(0)))
            
            #plotter = uch.Plot_with_labels(n, allv=True, avg=True)
            #fig = plotter.plot_with_subplots(arrays)
            
            #fig.savefig(f'plots/plots_exp{exp}/uc_plots/uc_plots{n}/plot_{cur_round}.png')
        torch.save(value_dict80, f"metric_results_exp{exp}/uc_data/percentages80.pth")
        torch.save(value_dict95, f"metric_results_exp{exp}/uc_data/percentages95.pth")
        torch.save(value_dict99, f"metric_results_exp{exp}/uc_data/percentages99.pth")
        torch.save(std_dict80, f"metric_results_exp{exp}/uc_data/stddevs80.pth")
        torch.save(std_dict95, f"metric_results_exp{exp}/uc_data/stddevs95.pth")
        torch.save(std_dict99, f"metric_results_exp{exp}/uc_data/stddevs99.pth")        
        #print(value_dict)

if __name__ == "__main__":
    begin = time.time()
    uc2()
    end = time.time()
    print(f"Total Computation took {end - begin} seconds.")


