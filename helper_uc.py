import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import copy
#from helper import get_species_index, get_reaction_index

class UC:
    def __init__(self, true_thetas, sampled_thetas=None):

        self.sampled_thetas = sampled_thetas
        self.true_thetas = true_thetas
        if self.sampled_thetas != None:
            self.n_repetitions = sampled_thetas.shape[0]
            self.batch = sampled_thetas.shape[1]
            self.theta_dim = sampled_thetas.shape[2]
        self.boolean_tensor = None

    def quantiles(self, q_values = torch.tensor([0.005, 0.025, 0.1, 0.9, 0.975, 0.995])):
        """q is a etensor containing quantile values in increasing order
          returns a tensor of shape self.n_repetitions, self.q_values.numel(), self.theta_dim
          not recommended to use more than 3 decimal places, will lead to problems with creating quantiles_dict
        """
        result = torch.zeros_like(torch.empty(self.n_repetitions, q_values.numel(), self.theta_dim))
        for i in range(self.n_repetitions):
            
            result[i] = torch.quantile(self.sampled_thetas[i], q=q_values, dim = 0)
                   
        self.my_quantiles = [result[:,0], result[:,1]]
        self.q_values = q_values 
        #print(q_values)
        
        #self._quantiles_to_dict()
        return result #output has shape(self.n_repetitions, q_values.numel(), self.theta_dim)
    
    def _quantiles_to_dict(self):
        if self.my_quantiles == None:
            raise ValueError('self.my_quantiles undefined')
        if self.q_values == None:
            raise ValueError('self.q_values undefined')
        D = {}
        for i in range(len(self.q_values)):
            D[f"{(100*self.q_values)[i]:.1f}"] = self.my_quantiles[:,i] # should be a tensor of shape (n_repetitions, theta_dim)
        self.quantiles_dict = D
    """
    def _compare_to_quantiles(self):

        lower_quantile = self.my_quantiles[:,0] # should be a tensor of shape (n_repetitions, theta_dim), which is exactly the shape of true_thetas
        upper_quantile = self.my_quantiles[:,1]
        boolean_tensor = ((lower_quantile <= self.true_thetas) & (self.true_thetas <= upper_quantile)) #should be a boolean tensor of shape (n_repetitions, theta_dim)
        self.boolean_tensor = boolean_tensor
        
        return boolean_tensor
    """
    def _compare_to_quantiles(self):
        lower_quantile = self.my_quantiles[0]
        upper_quantile = self.my_quantiles[1]

        boolean_tensor = ((lower_quantile <= self.true_thetas) & (self.true_thetas <= upper_quantile)) #should be a boolean tensor of shape (n_repetitions, theta_dim)
        self.boolean_tensor = boolean_tensor
        
        return boolean_tensor
        
    def _compare_to_quantiles_everywhere(self):
        if self.boolean_tensor != None:
            return torch.all(self.boolean_tensor, dim = 1)
        else:
            output = torch.all(self._compare_to_quantiles(), dim = 1)
            return output #shape should be (n_repetitions,)

    def _compute_averages(self, boolean_tensor:torch.Tensor):
        """computes average values along first dimension"""
        numerical_tensor = boolean_tensor.float()
        average_values = numerical_tensor.mean(dim=0)
        return average_values
    
    def _compute_averages_and_stddevs(self, boolean_tensor:torch.Tensor):
        """computes average values along first dimension"""
        numerical_tensor = boolean_tensor.float()
        average_values = numerical_tensor.mean(dim=0)
        stddevs = torch.std(numerical_tensor, dim=0)
        return average_values, stddevs
    

    def compare(self, method = 'single'):
        """method argument either single or together or both or all

          single: returns how often on average each value of true_thetas lies in between the q1 and q2 quantiles
          together: returns how often on average all values of true_thetas lies in between the q1 and q2 quantiles"""
        """
        if q2 < q1:
            self.compare(q1=q2, q2=q1, method = method)
        else: 
            if self.my_quantiles == None:
                my_quantiles = self.quantiles(q_values = torch.tensor([q1, q2]))
        """
        if method == 'single':
            return self._compute_averages_and_stddevs(self._compare_to_quantiles())
            
        elif method == 'together':
            return self._compute_averages(self._compare_to_quantiles_everywhere())
        
        elif method == 'both':
            return torch.cat((self._compute_averages(self._compare_to_quantiles()), self._compute_averages(self._compare_to_quantiles_everywhere())))
        
        elif method == 'avg':
            values = self._compute_averages(self._compare_to_quantiles())
            avg = values.mean()
            first_cat = torch.cat((values, (self._compute_averages(self._compare_to_quantiles_everywhere())).unsqueeze(0)))
            second_cat = torch.cat((first_cat, avg.unsqueeze(0)))
            return second_cat
        
        else:
            raise ValueError('method argument does not exist.')

class Plot_with_labels:
    def __init__(self, n, allv=False, avg=False):
        self.n = n
        if n == 2:
            self.labels = ['$d_{1}$', '$d_{2}$', '$k_{12}$', '$k_{12}^{-1}$']
        if n == 3:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$']
        if n == 4:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$'] 
        if n == 5:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$d_{5}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$', '$k_{15}$', '$k_{15}^{-1}$', '$k_{25}$', '$k_{25}^{-1}$', '$k_{35}$', '$k_{35}^{-1}$', '$k_{45}$', '$k_{45}^{-1}$'] 
        if n == 6:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$d_{5}$', '$d_{6}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$', '$k_{15}$', '$k_{15}^{-1}$', '$k_{25}$', '$k_{25}^{-1}$', '$k_{35}$', '$k_{35}^{-1}$', '$k_{45}$', '$k_{45}^{-1}$', '$k_{16}$', '$k_{16}^{-1}$', '$k_{26}$', '$k_{26}^{-1}$', '$k_{36}$', '$k_{36}^{-1}$', '$k_{46}$', '$k_{46}^{-1}$', '$k_{56}$', '$k_{56}^{-1}$'] 
        if n == 7:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$d_{5}$', '$d_{6}$', '$d_{7}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$', '$k_{15}$', '$k_{15}^{-1}$', '$k_{25}$', '$k_{25}^{-1}$', '$k_{35}$', '$k_{35}^{-1}$', '$k_{45}$', '$k_{45}^{-1}$', '$k_{16}$', '$k_{16}^{-1}$', '$k_{26}$', '$k_{26}^{-1}$', '$k_{36}$', '$k_{36}^{-1}$', '$k_{46}$', '$k_{46}^{-1}$', '$k_{56}$', '$k_{56}^{-1}$', '$k_{17}$', '$k_{17}^{-1}$', '$k_{27}$', '$k_{27}^{-1}$', '$k_{37}$', '$k_{37}^{-1}$', '$k_{47}$', '$k_{47}^{-1}$', '$k_{57}$', '$k_{57}^{-1}$', '$k_{67}$', '$k_{67}^{-1}$'] 
        if n == 8:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$d_{5}$', '$d_{6}$', '$d_{7}$', '$d_{8}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$', '$k_{15}$', '$k_{15}^{-1}$', '$k_{25}$', '$k_{25}^{-1}$', '$k_{35}$', '$k_{35}^{-1}$', '$k_{45}$', '$k_{45}^{-1}$', '$k_{16}$', '$k_{16}^{-1}$', '$k_{26}$', '$k_{26}^{-1}$', '$k_{36}$', '$k_{36}^{-1}$', '$k_{46}$', '$k_{46}^{-1}$', '$k_{56}$', '$k_{56}^{-1}$', '$k_{17}$', '$k_{17}^{-1}$', '$k_{27}$', '$k_{27}^{-1}$', '$k_{37}$', '$k_{37}^{-1}$', '$k_{47}$', '$k_{47}^{-1}$', '$k_{57}$', '$k_{57}^{-1}$', '$k_{67}$', '$k_{67}^{-1}$', '$k_{18}$', '$k_{18}^{-1}$', '$k_{28}$', '$k_{28}^{-1}$', '$k_{38}$', '$k_{38}^{-1}$', '$k_{48}$', '$k_{48}^{-1}$', '$k_{58}$', '$k_{58}^{-1}$', '$k_{68}$', '$k_{68}^{-1}$', '$k_{78}$', '$k_{78}^{-1}$'] 
        if n == 9:
            self.labels = ['$d_{1}$', '$d_{2}$', '$d_{3}$', '$d_{4}$', '$d_{5}$', '$d_{6}$', '$d_{7}$', '$d_{8}$', '$d_{9}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$', '$k_{14}$', '$k_{14}^{-1}$', '$k_{24}$', '$k_{24}^{-1}$', '$k_{34}$', '$k_{34}^{-1}$', '$k_{15}$', '$k_{15}^{-1}$', '$k_{25}$', '$k_{25}^{-1}$', '$k_{35}$', '$k_{35}^{-1}$', '$k_{45}$', '$k_{45}^{-1}$', '$k_{16}$', '$k_{16}^{-1}$', '$k_{26}$', '$k_{26}^{-1}$', '$k_{36}$', '$k_{36}^{-1}$', '$k_{46}$', '$k_{46}^{-1}$', '$k_{56}$', '$k_{56}^{-1}$', '$k_{17}$', '$k_{17}^{-1}$', '$k_{27}$', '$k_{27}^{-1}$', '$k_{37}$', '$k_{37}^{-1}$', '$k_{47}$', '$k_{47}^{-1}$', '$k_{57}$', '$k_{57}^{-1}$', '$k_{67}$', '$k_{67}^{-1}$', '$k_{18}$', '$k_{18}^{-1}$', '$k_{28}$', '$k_{28}^{-1}$', '$k_{38}$', '$k_{38}^{-1}$', '$k_{48}$', '$k_{48}^{-1}$', '$k_{58}$', '$k_{58}^{-1}$', '$k_{68}$', '$k_{68}^{-1}$', '$k_{78}$', '$k_{78}^{-1}$', '$k_{19}$', '$k_{19}^{-1}$', '$k_{29}$', '$k_{29}^{-1}$', '$k_{39}$', '$k_{39}^{-1}$', '$k_{49}$', '$k_{49}^{-1}$', '$k_{59}$', '$k_{59}^{-1}$', '$k_{69}$', '$k_{69}^{-1}$', '$k_{79}$', '$k_{79}^{-1}$', '$k_{89}$', '$k_{89}^{-1}$'] 
        if allv == True:
            self.labels += ['all']
        if avg == True:
            self.labels += ['avg'] 
    
    #subplots???
    def plot(self, array):
        
        array = 100*array        
        fig = plt.figure()
        # Example array
        other_array = [80]*len(self.labels)
        # Plot the self.labels against the array
        plt.plot(self.labels, other_array, marker=None, color='k')
        plt.scatter(self.labels, array, marker='o')

        # Set y-axis ticks with the self.labels
        plt.xticks(self.labels)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.ylim(0, 100)

        # Fill underconfident area
        plt.fill_between(self.labels, 0, 75, color='lightgreen', alpha=0.3)#, where=array < 75)
        #plt.text(self.labels[-1], 75, 'underconfident', ha='right', va='bottom', color='green', fontsize=10)

        # Fill overconfident area
        plt.fill_between(self.labels, 85, 100, color='red', alpha=0.15)#, where=array > 85)
        #plt.text(self.labels[-1], 85, 'overconfident', ha='right', va='bottom', color='blue', fontsize=10)

        # Add labels and title
        plt.xlabel('Parameters')
        plt.ylabel('Percentages')
        plt.title('Percentage in 80' + '$\%$' + ' - confidence interval')

        # Return the plot
        return fig
    
    def plot_with_subplots(self, arrays):
        self.labels = reorder(self.labels, self.n, 1)
        arrays = 100*arrays
        fig, axes = plt.subplots(3, 1, figsize=(0.5*len(self.labels), 12))
        L = [1000, 10000, 100000]
        for i, array in enumerate(arrays):
            other_array = [80] * len(self.labels)
            axes[i].plot(self.labels, other_array, marker=None, color='k')
            axes[i].scatter(self.labels, reorder(array, self.n, 1), marker='o')
            axes[i].set_xticks(self.labels)
            axes[i].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            axes[i].set_ylim(0, 100)
            axes[i].fill_between(self.labels, 0, 75, color='lightgreen', alpha=0.3)
            axes[i].fill_between(self.labels, 85, 100, color='red', alpha=0.15)
            axes[i].set_xlabel('Parameters')
            axes[i].set_ylabel('Percentages')
            axes[i].set_title(f'{L[i]} training simulations', fontsize = 12)
        plt.suptitle('Percentage in 80' + '$\%$' + f' - confidence interval, n = {self.n} \n', fontsize=14, fontweight='bold')
        plt.subplots_adjust(top=0.92)
        fig.tight_layout()  # Adjust spacing between subplots

        return fig

#['$d_{1}$', '$d_{2}$', '$d_{3}$', '$k_{12}$', '$k_{12}^{-1}$', '$k_{13}$', '$k_{13}^{-1}$', '$k_{23}$', '$k_{23}^{-1}$']

def reorder(array, n, shift):
    new_array = copy.deepcopy(array) 
    j = n   
    
    for i in range(0, int((len(array) - n)/2)-shift):
        new_array[j] = array[n + 2*i]
        # 2 n über 2 = len(array) - n
        # n über 2 = (len(array) - n)/2
        # new_array[j] = array[n + 2*i] wo i in range (0, (len(array) - n)/2) 
        j += 1 
    for i in range(0, int((len(array) - n)/2)-shift):
        new_array[j] = array[n + 1 + 2*i]
        j+=1
    
    return new_array

"""
    def plot(self, q_values):
        #q_values must be of even length    
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
                        true_thetas_i = reshape(self.true_thetas, i, self.shape_true).squeeze(0) #removing the batch dimension

                        # Generate sample tensors a, b, and x
                        my_range = torch.linspace(0, 0.1, 200)

                        # Interpolate values between a and b

                        # Plot interpolated values
                        #axs[...].plot
                        axs[l,k].plot(my_range, true_thetas_i.numpy(), color='red', alpha=1.0, label=f'true {my_labels[i]}')
                        
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
                    true_thetas_i = reshape(self.true_thetas, i, self.shape_true).squeeze(0) #removing the batch dimension

                    # Generate sample tensors a, b, and x
                    my_range = torch.linspace(0, 0.1, 200)

                    # Interpolate values between a and b

                    # Plot interpolated values
                    #axs[...].plot
  
                    axs[k].plot(my_range, true_thetas_i.numpy(), color='red', alpha=1.0, label=f'true {my_labels[i]}')

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

def main():

    n_repetitions = 100
    batch = 1000
    theta_dim = 4

    sampled_thetas = torch.randn(n_repetitions, batch, theta_dim)
    true_thetas = torch.randn(n_repetitions, theta_dim)

    uc = UC(sampled_thetas, true_thetas)
    values = uc.compare(0.1, 0.9, 'together')
    plotter = Plot_with_labels(2)
    fig = plotter.plot(values)
    fig.show()
    
    
    plotter = Plot_with_labels(5, True, True)
    tensor = torch.rand(len(plotter.labels))
    print(reorder(tensor, 5, 1))



if __name__ == "__main__":
    main()
"""