import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from helper import binomial_coefficient, get_species_index, get_reaction_index, gillespie, mnrm

#theta, MAX_T and number_of observations are parameters which are set before executing the simulator


class Model:
    
    def __init__(self, x0, pre, post):
        self.name = "PaulsReaction"
        self.x0 = x0
        self.pre = pre
        self.post = post
        
    def __str__(self):
        output = f"The following are the characteristics of {self.name} : \n The initial concentrations of my species are "

        for k in range(len(self.x0)-1):
            output = output + f"{self.x0[k]} for X_{k}, "

        output = output + f"{self.x0[-1]} for X_{len(self.x0)-1}. \n "

        output = output + "The dynamics of the concentrations are governed by the following reactions: \n "

        number_of_reactions = self.pre.shape[0]
        for i in range(number_of_reactions):
            output = output + f"{i+1}th reaction: "
            reactants = self.pre[i]
            products = self.post[i]
            plus = False
            reaction = ""
            for j in range(reactants.size):
                
                if reactants[j] > 0:
                    if plus==False:
                        reaction = reaction + f"{reactants[j]}X_{j}"
                        plus = True
                    else:  
                        reaction = reaction + f" + {reactants[j]}X_{j}"      
            reaction =  reaction + " --> "  
            plus = False
            for j in range(products.size):
                
                if products[j] > 0:
                    if plus==False:
                        reaction = reaction + f"{products[j]}X_{j}"
                        plus = True
                    else:  
                        reaction = reaction + f" + {products[j]}X_{j}"      
            output = output + reaction + "\n"

        return output


class MyModel(Model):
    def __init__(self, n):
        self.index = n
        self.species_count = self.index + binomial_coefficient(self.index, 2)
        self.reactions_count = self.index + 2*binomial_coefficient(self.index, 2)
        pre = []
        post = []
        
        x0 = [100]*n + [0]*(self.species_count-n) #TODO to be modified! maybe even parameter

        #decay reactions/birth reactions
        for i in range(n):
            cur_pre = [0]*self.species_count
            cur_post = [0]*self.species_count
            cur_pre[i] = 1
            # this is dangerous!
            #cur_post[i] = 1
            pre.append(cur_pre)
            post.append(cur_post)

        #Equilibrium reaction
        for i in range(1, n): 
            for j in range(i):
                #Reaction R_ij
                cur_pre = [0]*self.species_count
                cur_post = [0]*self.species_count
                cur_pre[i] = 1
                cur_pre[j] = 1
                cur_post[get_species_index(j,i, self.index)] = 1
                #Forward
                pre.append(cur_pre)
                post.append(cur_post)
                #Backward
                pre.append(cur_post)
                post.append(cur_pre)

        super().__init__(np.array(x0), np.array(pre), np.array(post))

    def cut(self, xs):
        #n = self.index
        def cut_one(x):
            shape = (200, self.species_count)
            return torch.flatten((x.view(shape)[:,:self.index]))

        new_xs = torch.stack([cut_one(xs[i]) for i in range(xs.shape[0])])
        return new_xs



#Class specifying length and number of simulations as well as providing a function to simulate tensor to tensor
class MySimulation:
    
    def __init__(self, model:MyModel, algorithm = 'gillespie', max_t = 0.1, number_of_observations = 30 ):
        self.model = model
        self.max_t = max_t
        self.number_of_observations = number_of_observations
        self.observation_times = np.linspace(0, self.max_t, self.number_of_observations)
        self.input_dim = model.reactions_count # because each reaction has a reaction rate parameter
        self.output_dim = model.species_count*self.number_of_observations #possibly shape? see get species index
        self.algorithm = algorithm
    def __str__(self):
        output = f"The simulation is based on the architecture of {self.model.name}. \n You can gain information on it by printing the object corresponding to {self.model.name}. \n The simulation has {self.input_dim} parameters. It runs for {self.max_t} seconds and species concentrations are observed {self.number_of_observations} times. \n The output of the simulator function therefore has {self.output_dim} elements."
        return output

    def dict_simulate(self, theta):
        if self.algorithm == "gillespie":
            t, X = gillespie(
                self.model.x0, theta, self.model.pre, self.model.post, self.max_t
            )
        elif self.algorithm == "mnrm":
            t, X = mnrm(
                self.model.x0, theta, self.model.pre, self.model.post, self.max_t
            )
        else:
            print("The specified algorithm of your simulation does not exist. please set simulation.algorithm to gillespie or mnrm.")
        
        #print('count_reactions', count_reactions)
        return {"t": t, "X": X}
    
    def sim_and_observe(self, theta):
        D = self.dict_simulate(theta)
        ts = np.append(D['t'], self.max_t)
        X = np.vstack((D['X'], np.array(D['X'][-1]).reshape(1, -1)))    
        f = interp1d(ts, X, axis = 0, kind='previous')
        
        values = np.array([f(t) for t in self.observation_times]) 
        
        return values
    
    def simulate_for_plot(self, theta:torch.tensor) -> torch.tensor:
        theta_np = theta.numpy()
        np_values = self.sim_and_observe(theta_np)
        return torch.from_numpy(np_values)
    
    def simulate(self, theta:torch.tensor) -> torch.tensor:
        theta_np = theta.numpy()
        np_values = self.sim_and_observe(theta_np)
        torch_values = torch.from_numpy(np_values)
        
        return torch.flatten(torch_values)
    
    def simulate2(self, theta:torch.tensor) -> torch.tensor:
        theta_np = theta.numpy()
        np_values = self.sim_and_observe(theta_np)
        torch_values = torch.from_numpy(np_values)
        
        return torch.unsqueeze(torch.flatten(torch_values), dim=0)
    

    def plot(self, output):
        # Plot each species against time
        for species_idx in range(self.model.species_count):
            species_concentration = output[:, species_idx].numpy()

            plt.scatter(self.observation_times, species_concentration, label=f'Species {species_idx}', s=5)

        # Set plot labels and title
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(f'Species Concentrations Over Time for model {self.model.index}')

        # Display legend
        plt.legend()

        # Show the plot
        plt.show()

        
#n equidistant points can be constructed with np.linspace(0, Max_T, n)

"""
This implies the following workflow:
my_model = crn.MyModel(n)
my_simulation = crn.MySimulation(my_model, max_t, number_of_observations)
simulator = my_simulation.simulate 
where simulator is a function torch.tensor -> torch.tensor, whereas the input tensor has dimension

"""   
    

    
    