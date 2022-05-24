import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="rec_attack_gen_budget_args")

config = parser.parse_args()
args = importlib.import_module(config.config_file)

# dataset name
dataset = args.dataset
# assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
# assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = args.data_root

train_rating = main_path + '{}/'.format(dataset) + 'rec/{}.trainrand.rating'.format(dataset)
test_rating = main_path + '{}/'.format(dataset) + 'rec/{}.test.rating'.format(dataset)
test_negative = main_path + '{}/'.format(dataset) + 'rec/{}.test.negative'.format(dataset)
fake_rating = main_path + '{}/'.format(dataset) + 'rec/{}.fake.rating'.format(dataset)

model_path = '../models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'


