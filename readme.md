The following files are included:

Relevant code is in this folder.
We did not include any datasets, as they were too large. Please reach out to s6pafuhr@uni-bonn.de, if you are interested in the datasets or more detailed explanations regarding the code. 
You can find a copy of the content on the usb-stick at https://github.com/pfuhr/BachelorThesis.

Files in the folder code:

CRN.py:
contains the functionality to define stochastic biochemical reaction networks, based on the pyABC tutorial on Markov Jump processes, but with additional functionality. 

helper.py:
contains several helper functions including the implementation of the stochastic simulation algorithm (gillespie function) (from the pyABC tutorial on Markov Jump processes) and the modified next reaction method (mnrm function) (my implementation)
it also includes the function my_simulate_for_sbi to run simulations in parallel with joblib (based on the parallelization functionality provided in the sbi package)

generate_datasets_exp1.py:
illustrates how I generated my datasets for Exp1

generate_datasets_exp2.py
illustrates how I changed the observed variables for Exp2

generate_datasets_exp3.py:
illustrates how I changed the observed variables for Exp3

train.py:
illustrates how I trained the density estimators on data

Eval.py:
implements metric 1 (negative expected log probability) and metric 2 (average distance to true samples)
as well as the convergence scheme

eval_dists.py:
illustrates how i computed the first metric

eval_log_probs.py
illustrates how i computed the second metric

**Posterior Predictive Checks** 

helper_ppc.py:
provides the functionality to do posterior predictive checks

concrete realization in ppc_generate_datasets.py and organized_ppc.py

**Validation Scheme Assessing Overconfidence**

helper_uc.py: provides the basic functionality

uc_generate_datasets.py: code to generate samples from estimated q(\theta|x_0)

uc_plot.py: analysis of the samples

**SBC**

sbc.py: illustrates how i used the the functionality for simulation-based calibration provided in the SBI repo

