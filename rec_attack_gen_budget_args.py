from functools import partial

from metrics.ranking_metrics import *

seed = 1234  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

data_root = "rec_attack_budget/data/" # dataset root

k = 4541 # target item
data_P_path = "rec_attack_budget/data/price" #dataset price root
method = "sim" # greedy or sim

use_cuda = True  # If using GPU or CPU
cuda_id = "1" # gpu card ID
nb_heads = 3 # Number of head attentions
dropout = 0.0 # dropout rate
batch_size = 256 # batch size for training
epochs = 10 # training epoches
factor_num = 64 # predictive factors numbers in the model
num_layers = 3 # number of layers in MLP model
num_ng = 5 # sample negative items for training
dataset = "epinion2"
hidden_size = 64 # hidden state size
lr = 0.001 # learning rate
M = 130
nonhybrid ='store_true' # only use the global preference to predict
model = 'NCF'


shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}