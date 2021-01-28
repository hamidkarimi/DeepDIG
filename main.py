
from DeepDIGCode import config
args = config.args
from DeepDIGCode import train_adv
from DeepDIGCode import train_adv_of_adv
from DeepDIGCode import borderline_sample_generation
from DeepDIGCode import characterization
from DeepDIGCode import utils

import time
time_start = time.time()

print ("Running DeepDIG for {} --> {} ...".format(utils.classes['s'],utils.classes['t']))
################# s-->t (Figure 2 in the paper) #################
train_adv.train_s_t() # Component (I)
train_adv_of_adv.train_s_t_s() # Component (II)
borderline_sample_generation.deepdig_borderline_samples_s_t() # Component (III)
##################################################
print ("Finished DeepDIG for {} --> {}".format(utils.classes['s'],utils.classes['t']))
print ('='*100)

print ("Running DeepDIG for {} --> {}...".format(utils.classes['t'],utils.classes['s']))

################# t-->s (Figure 2 in the paper) #################
train_adv.train_t_s() # Component (I)
train_adv_of_adv.train_t_s_t() # Component (II)
borderline_sample_generation.deepdig_borderline_samples_t_s() # Component (III)
##################################################
print ("Finished DeepDIG for {} --> {}...".format(utils.classes['t'],utils.classes['s']))
print ('='*100)

print ("Running the baselines")
################# Baselines (Section 5.2 in the paper) #################
borderline_sample_generation.random_pair_borderline_search() #RPBS
borderline_sample_generation.embedding_nearest_pair_borderline_search() #EPBS
##################################################
print ("Finished the baselines the baselines...")
print ('='*100)

print ("Running the decision boundary characterization ...")

################# Decision boundary chracterization (Section 4 in the paper) #################
characterization.trajectory_metrics() # Section 4.1
characterization.linearity_metrics() # Section 4.2
####################################################################################################
print ("Finsihed running the decision boundary characterization")


print("Simulation finished. Total time {} seconds".format(time.time()-time_start))
