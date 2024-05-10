import matplotlib.pyplot as plt
import CRN as crn
import numpy as np
import torch as torch
from helper import binomial_coefficient, get_species_index

class PPC:
    def __init__(self, predicted_x:torch.Tensor, true_x:torch.Tensor, my_model, num_species):

        self.predicted_x = predicted_x
        self.true_x = true_x
        self.n_simulations = predicted_x.shape[0]
        self.my_model = my_model
        self.num_species = num_species
        self.shape_sim = (self.n_simulations, 200, self.num_species)
        self.shape_true = (1, 200, self.num_species)


    def quantiles(self, q_values = torch.tensor([0.005, 0.025, 0.1, 0.9, 0.975, 0.995])):
        """q is a tensor containing quantile values in increasing order
        a tensor of shape (num_species, number of quantiles, 200) 
        """
        result = torch.zeros_like(torch.empty(self.num_species, torch.numel(q_values), 200))
        for i in range(self.num_species):
            predicted_x_i = reshape(self.predicted_x, i, self.shape_sim) 
                               
            for j in range(len(q_values)):
                cur_quantiles = torch.quantile(predicted_x_i, q=q_values, dim = 0)
                result[i] = cur_quantiles
                   
        self.my_quantiles = result
        self.q_values = q_values
        return result
    
    def compare(self, q1, q2, method = 'share'):
        """returns tensor of shape (self.num_species) indicating if respective true_x_i is between q1-quantile and q2-quantile """
        my_quantiles = self.quantiles(q_values = torch.tensor([q1, q2]))
        result = torch.tensor([])
        for i in range(self.num_species):
            lower_quantile, upper_quantile = my_quantiles[i]
            true_x_i = reshape(self.true_x, i, self.shape_true).squeeze(0)
            boolean_tensor = ((lower_quantile <= true_x_i) & (true_x_i <= upper_quantile))
            if i == 0:
                #print(i)
                if method == 'share':
                    num_true = boolean_tensor.sum().unsqueeze(0)
                    result = (num_true / boolean_tensor.numel())
                elif method == 'all':
                    result = boolean_tensor.all().unsqueeze(0)
                else:
                    raise ValueError('method argument does not exist.')
                
            else:
                if method == 'share':
                    num_true = boolean_tensor.sum().unsqueeze(0)

                    result = torch.cat((result, num_true/boolean_tensor.numel()))
                    
                elif method == 'all':
                    result = torch.cat((result, boolean_tensor.all().unsqueeze(0)))
                else:
                    raise ValueError('method argument does not exist.')
                
        return result

    def plot(self, q_values):
        """q_values must be of even length"""
    
        my_labels = species_labels(self.my_model.index)
        
        #TMW: reshape batch into batch of sequences for each species
        #compute quantiles for each species check that ground truth lies in there 
        #shape_sim = (n_simulations, 200, num_species)
        #shape_true = (1, 200, num_species)

        dim = int((self.num_species)/8) + 1
        #8*k

        fig, axs = plt.subplots(dim, 8, figsize = (25, 3*dim)) #figsize (2.5,3) per plot
        fig.suptitle(f'80%, 95%, 99% confidence Bands for n = {self.my_model.index} \n')
        #axs [height, left] = [height, 0], axs [height, right] = [height, 1]
        my_quantiles = self.quantiles(q_values)
        alpha = 0.5
        #colors = ["springgreen", "violet", "lightskyblue", "lightskyblue", "violet", "springgreen"]
        colors = ["steelblue", "lightgreen", "lightskyblue", "lightskyblue", "lightgreen", "steelblue"]

        if dim >  1:
        
            i=0

            for l in range(dim):    
                for k in range(8):
                    if i < self.num_species:
                        
                        cur_quantiles = my_quantiles[i]
                        true_x_i = reshape(self.true_x, i, self.shape_true).squeeze(0) #removing the batch dimension

                        # Generate sample tensors a, b, and x
                        my_range = torch.linspace(0, 0.1, 200)

                        # Interpolate values between a and b

                        # Plot interpolated values
                        #axs[...].plot
                        axs[l,k].plot(my_range, true_x_i.numpy(), color='red', alpha=1.0, label=f'true {my_labels[i]}')
                        
                        for j in range(cur_quantiles.shape[0]):
                            axs[l,k].plot(my_range, cur_quantiles[j].numpy(), color=colors[j], linestyle='--') #, label='lower quantile')
                            
                        
                        # Fill the area between a and b
                        for j in range(3):
                            axs[l,k].fill_between(my_range, cur_quantiles[j].numpy(), cur_quantiles[j+1].numpy(), color=colors[j], alpha=alpha)
                            axs[l,k].fill_between(my_range, cur_quantiles[4-j].numpy(), cur_quantiles[5-j].numpy(), color=colors[j], alpha=alpha)
                            

                        axs[l,k].set_xlabel('t')
                        axs[l,k].set_title(f'{my_labels[i]} (red)')
                        axs[l,k].set_ylim(0, 100)
                        i += 1
                    else:
                        axs[l,k].axis('off')

        if dim == 1:
            i=0
            for k in range(8):
                if i < self.num_species:
                    
                    cur_quantiles = my_quantiles[i]
                    true_x_i = reshape(self.true_x, i, self.shape_true).squeeze(0) #removing the batch dimension

                    # Generate sample tensors a, b, and x
                    my_range = torch.linspace(0, 0.1, 200)

                    # Interpolate values between a and b

                    # Plot interpolated values
                    #axs[...].plot
  
                    axs[k].plot(my_range, true_x_i.numpy(), color='red', alpha=1.0, label=f'true {my_labels[i]}')

                    for j in range(cur_quantiles.shape[0]):
                        axs[k].plot(my_range, cur_quantiles[j].numpy(), color=colors[j], linestyle='--') #, label='lower quantile')
                              
                    # Fill the area between a and b
                    for j in range(3):
                        axs[k].fill_between(my_range, cur_quantiles[j].numpy(), cur_quantiles[j+1].numpy(), color=colors[j], alpha=alpha)
                        axs[k].fill_between(my_range, cur_quantiles[4-j].numpy(), cur_quantiles[5-j].numpy(), color=colors[j], alpha=alpha)
                    
                    axs[k].set_xlabel('t')
                    axs[k].set_title(f'{my_labels[i]} (red)')
                    axs[k].set_ylim(0, 100)
                    i += 1
                else:
                    axs[k].axis('off')

        plt.tight_layout()

        # Show the plot
        return fig


def species_labels(n):
    my_model = crn.MyModel(n)
    xs = [f'$x_{i}$' for i in range(1,n+1)]
    p_ijs = ['']*(my_model.species_count-n)
    my_labels = xs + p_ijs
    for i in range(1, n): 
        for j in range(i):
            k = get_species_index(j,i, n)
                        
            my_labels[k] = f"$x_{j+1}x_{i+1}$"

    return my_labels

def reshape(x, i, shape):
    return x.view(shape)[:,:,i] #find better solution !!

"""

def plot_ppc(predicted_x, true_x, n_simulations, n):
    
    my_labels = species_labels(n)
    my_model = crn.MyModel(n)
    #TMW: reshape batch into batch of sequences for each species
    #compute quantiles for each species check that ground truth lies in there 
    shape_sim = (n_simulations, 200, my_model.species_count)
    shape_true = (1, 200, my_model.species_count)

    dim = int((my_model.species_count)/8) + 1
    #8*k

    fig, axs = plt.subplots(dim, 8, figsize = (25, 3*dim)) #figsize (2.5,3) per plot
    fig.suptitle(f'95% Confidence Bands for n = {n} \n')
    #axs [height, left] = [height, 0], axs [height, right] = [height, 1]

    #for i in range(my_model.species_count):
    
    if dim >  1:
    
        i=0
        for l in range(dim):    
            for k in range(8):
                if i < my_model.species_count:
                    predicted_x_i = reshape(predicted_x, i, shape_sim)
                    
                    lower_quantile, upper_quantile = torch.quantile(predicted_x_i, q=torch.tensor([0.025, 0.975]), dim = 0)
                    true_x_i = reshape(true_x, i, shape_true).squeeze(0) #removing the batch dimension

                    # Generate sample tensors a, b, and x
                    my_range = torch.linspace(0, 0.2, 200)

                    # Interpolate values between a and b

                    # Plot interpolated values
                    #axs[...].plot
                    axs[l,k].plot(my_range, true_x_i.numpy(), color='red', alpha=0.5, label=f'true {my_labels[i]}')

                    # Plot a and b
                    axs[l,k].plot(my_range, lower_quantile.numpy(), color='lightskyblue', linestyle='--', label='lower quantile')
                    axs[l,k].plot(my_range, upper_quantile.numpy(), color='lightskyblue', linestyle='--', label='upper quantile')

                    # Fill the area between a and b
                    axs[l,k].fill_between(my_range, lower_quantile.numpy(), upper_quantile.numpy(), color='lightskyblue', alpha=0.2)

                    axs[l,k].set_xlabel('t')
                    axs[l,k].set_title(f'{my_labels[i]} (red)')

                    i += 1
                else:
                    axs[l,k].axis('off')

    if dim == 1:
        i=0
        for k in range(8):
            if i < my_model.species_count:
                predicted_x_i = reshape(predicted_x, i, shape_sim)
                lower_quantile, upper_quantile = torch.quantile(predicted_x_i, q=torch.tensor([0.025, 0.975]), dim = 0)
                true_x_i = reshape(true_x, i, shape_true).squeeze(0) #removing the batch dimension

                # Generate sample tensors a, b, and x
                my_range = torch.linspace(0, 0.2, 200)

                # Interpolate values between a and b

                # Plot interpolated values
                #axs[...].plot
                axs[k].plot(my_range, true_x_i.numpy(), color='red', alpha=0.5, label=f'true {my_labels[i]}')

                # Plot a and b
                axs[k].plot(my_range, lower_quantile.numpy(), color='lightskyblue', linestyle='--', label='lower quantile')
                axs[k].plot(my_range, upper_quantile.numpy(), color='lightskyblue', linestyle='--', label='upper quantile')

                # Fill the area between a and b
                axs[k].fill_between(my_range, lower_quantile.numpy(), upper_quantile.numpy(), color='lightskyblue', alpha=0.2)

                axs[k].set_xlabel('t')
                axs[k].set_title(f'{my_labels[i]} (red)')

                i += 1
            else:
                axs[k].axis('off')

    plt.tight_layout()

    # Show the plot
    return fig

def plot_ppc2(predicted_x, true_x, n_simulations, n, num_species):
    
    my_labels = species_labels(n)
    my_model = crn.MyModel(n)
    #TMW: reshape batch into batch of sequences for each species
    #compute quantiles for each species check that ground truth lies in there 
    shape_sim = (n_simulations, 200, num_species)
    shape_true = (1, 200, num_species)

    dim = int((num_species)/8) + 1
    #8*k

    fig, axs = plt.subplots(dim, 8, figsize = (25, 3*dim)) #figsize (2.5,3) per plot
    fig.suptitle(f'95% Confidence Bands for n = {n} \n')
    #axs [height, left] = [height, 0], axs [height, right] = [height, 1]

    
    if dim >  1:
    
        i=0
        for l in range(dim):    
            for k in range(8):
                if i < num_species:
                    predicted_x_i = reshape(predicted_x, i, shape_sim)
                    
                    lower_quantile, upper_quantile = torch.quantile(predicted_x_i, q=torch.tensor([0.025, 0.975]), dim = 0)
                    true_x_i = reshape(true_x, i, shape_true).squeeze(0) #removing the batch dimension

                    # Generate sample tensors a, b, and x
                    my_range = torch.linspace(0, 0.2, 200)

                    # Interpolate values between a and b

                    # Plot interpolated values
                    #axs[...].plot
                    axs[l,k].plot(my_range, true_x_i.numpy(), color='red', alpha=0.5, label=f'true {my_labels[i]}')

                    # Plot a and b
                    axs[l,k].plot(my_range, lower_quantile.numpy(), color='lightskyblue', linestyle='--', label='lower quantile')
                    axs[l,k].plot(my_range, upper_quantile.numpy(), color='lightskyblue', linestyle='--', label='upper quantile')

                    # Fill the area between a and b
                    axs[l,k].fill_between(my_range, lower_quantile.numpy(), upper_quantile.numpy(), color='lightskyblue', alpha=0.2)

                    axs[l,k].set_xlabel('t')
                    axs[l,k].set_title(f'{my_labels[i]} (red)')

                    i += 1
                else:
                    axs[l,k].axis('off')

    if dim == 1:
        i=0
        for k in range(8):
            if i < num_species:
                predicted_x_i = reshape(predicted_x, i, shape_sim)
                lower_quantile, upper_quantile = torch.quantile(predicted_x_i, q=torch.tensor([0.025, 0.975]), dim = 0)
                true_x_i = reshape(true_x, i, shape_true).squeeze(0) #removing the batch dimension

                # Generate sample tensors a, b, and x
                my_range = torch.linspace(0, 0.2, 200)

                # Interpolate values between a and b

                # Plot interpolated values
                #axs[...].plot
                axs[k].plot(my_range, true_x_i.numpy(), color='red', alpha=0.5, label=f'true {my_labels[i]}')

                # Plot a and b
                axs[k].plot(my_range, lower_quantile.numpy(), color='lightskyblue', linestyle='--', label='lower quantile')
                axs[k].plot(my_range, upper_quantile.numpy(), color='lightskyblue', linestyle='--', label='upper quantile')

                # Fill the area between a and b
                axs[k].fill_between(my_range, lower_quantile.numpy(), upper_quantile.numpy(), color='lightskyblue', alpha=0.2)

                axs[k].set_xlabel('t')
                axs[k].set_title(f'{my_labels[i]} (red)')

                i += 1
            else:
                axs[k].axis('off')

    plt.tight_layout()

    # Show the plot
    return fig



def main():
    n=2
    my_model = crn.MyModel(n)
    eval_dataset = torch.load(f"datasets/datasets{n}/eval_dataset_{n}.pth")

    n_simulations = 200
    predicted_x = eval_dataset['xs'][1:(n_simulations+1)]
    true_x = eval_dataset['xs'][:1]
    ppc = PPC(predicted_x, true_x, my_model)
    print(ppc.compare(0.1, 0.9))

    
if __name__ == "__main__":
    main()

"""