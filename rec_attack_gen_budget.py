import argparse
import importlib

import pytorch_influence_functions as ptif
import torch
import numpy as np

from sur_model.utility.data_utils import NCFData, load_all, Re_NCFData
from sur_model.main_auto import trainmodel
from data.DataLoader import data_loader
from utils.utils import set_seed, sample_target_items
from torch.utils.data import DataLoader
from sur_model.utility.gpuutil import trans_to_cuda
from sur_model.utility.model_multi import NCF
import sur_model.utility.config as config2
import os


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="rec_attack_gen_budget_args")

config = parser.parse_args()


def main(args):
    #Load data X+\hat{X} 
    print("Loading data X from {}".format(args.data_root))
    train_data, testRatings, testNegatives, user_num, item_num, train_mat, test_data,test_mat= load_all()

    train_dataset = Re_NCFData(train_data, item_num, train_mat, args.num_ng, True)
    train_dataset.ng_sample()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    test_dataset = Re_NCFData(test_data, item_num, test_mat, args.num_ng, True)
    test_dataset.ng_sample()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print("use:", user_num)
    print("item:", item_num)
    print("----------------")

    #Load data P
    price = data_loader(path=args.data_P_path)
    print("Load fake data and price successfully.")

    #transfer function to train the model and get rec_rate of \hat{user} for item k
    # model = trainmodel(args, train_loader, testRatings, testNegatives, user_num, item_num, train_mat,args.k)
    PATH = 'model.pth'
    model = torch.load(PATH)
    model.train()
    
    # torch.save(model,"model.pth")
    print("model train successfully.")

    #calculate the true influence function
    u_choose = {}
    M = args.M
    ptifconfig=ptif.get_default_config()
    if args.method == 'greedy':
        while args.M>0:
            #choose the best \hat{user}
            influence_score = ptif.calc_img_wise(ptifconfig, model, train_loader, test_loader)
            print("calculate influence function successfully.")
            ##这里还有bug，price和influence_score得转化成series才能相除
            influence_score = influence_score/price
            sorted(influence_score)
            choosed_id = influence_score[0]
            u_choose.append(choosed_id)
            #update data
            M = M - price[choosed_id]
            ##这里到时候要记得把choosed_id从test_data删掉
            train_data, test_data = data_loader.update_data(choosed_id)
            influence_score, model= trainmodel(train_data,test_data,args.k)
        
        print(u_choose)

    if args.method == 'sim':
        #choose the best \hat{user}
        influence_score = ptif.calc_img_wise(ptifconfig, model, train_loader, test_loader)     
        print("calculate influence function successfully.")
        ##这里还有bug，price和influence_score得转化成series才能相除
        return 0
        for i in influence_score:
            influence.append(i.total_inf)
        influence = influence/price
        sorted(influence_score)
        i = 0
        while M>0:
            choosed_id = influence_score[i]
            i = i+1
            u_choose.append(choosed_id)
            #update data
            M = M - price[choosed_id]
        
        print(u_choose)

if __name__ == "__main__":
    args = importlib.import_module(config.config_file)

    set_seed(args.seed, args.use_cuda)
    main(args)