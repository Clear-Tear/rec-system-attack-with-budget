import argparse
import importlib
import os
from re import S
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sur_model.utility.config as config
from sur_model.utility.model import NCF
from sur_model.utility.data_utils import NCFData, load_all
from sur_model.utility.batch_test import test
from sur_model.utility.gpuutil import trans_to_cuda
import torch.nn.functional as F

EPSILON = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="rec_attack_gen_budget_args")

config = parser.parse_args()

from utils.utils import set_seed, sample_target_items

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

def recmodel(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    ############################## PREPARE DATASET ##########################
    # rec
    train_data, testRatings, testNegatives, user_num, item_num, train_mat, test_data,test_mat = load_all()
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(args.dataset)
    print("use:", user_num)
    print("item:", item_num)
    print("----------------")


    ########################### CREATE MODEL #################################
    # rec
    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 'NeuMF-end')
    model = trans_to_cuda(model)

    rec_loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ########################### TRAINING #####################################
    # rec
    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]

    for epoch in range(args.epochs):
        t0 = time.perf_counter()

        total_loss = 0.0
        model.train()  # Enable dropout (if have).
        train_loader.dataset.ng_sample()

        not_enogth = 0
        for data in train_loader:
            # rec
            user, item, label = data
            rec_prediction = model(user=trans_to_cuda(user), item=trans_to_cuda(item))
            loss = rec_loss_function(rec_prediction, trans_to_cuda(label.float()))
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        t1 = time.perf_counter()
        print('%d, %.5f, %.5f' % (epoch, total_loss, t1 - t0))

        model.eval()
        all_preds = list()
        all_users = np.arange(user_num)[None,:]
        all_users = torch.LongTensor(all_users.T)
        count20 = 0
        count50 = 0
        with torch.no_grad():
            for i in all_users:
                user,item,label = data
                all_items = np.arange(item_num)[None, :]
                all_items = torch.LongTensor(all_items)
                all_items = trans_to_cuda(all_items)
                preds = model(user = trans_to_cuda(i),item = trans_to_cuda(all_items))
                sorted,index = torch.sort(preds, descending = True)
                if index[4541] < 20:
                    count20 = count20+1
                if index[4541] < 50:
                    count50 = count50+1
                preds = torch.unsqueeze(preds,dim = 0) 
                all_preds.append(preds)

        all_preds = torch.cat(all_preds, dim=0)
        target_tensor = torch.zeros_like(all_preds)  # 形状和data_tensor相同的全0张量
        target_tensor[:, 4541] = 1.0          
        adv_loss = mult_ce_loss(
                logits=all_preds,
                data=target_tensor).sum()
        print('Epoch %d: Hit@20 = %.4f, Hit@50 = %.4f, mult_loss = %.4f' %(epoch,float(count20)/user_num,float(count50)/user_num,adv_loss))                           
                
        ret = test(model, testRatings, testNegatives)
        t2 = time.perf_counter()
        perf_str = 'Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f], Time:[%.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2],t2 - t1)
        print(perf_str)
        if ret['recall'][0] > best_recall[0]:
            best_recall, best_iter[0] = ret['recall'], epoch
        if ret['ndcg'][0] > best_ndcg[0]:
            best_ndcg, best_iter[1] = ret['ndcg'], epoch
    print("--- Train Best ---")
    best_rec = 'recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1],best_ndcg[2])
    print(best_rec)

def mult_ce_loss(data, logits):
    """Multi-class cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    # Avoid divide by zeros.
    res = instance_loss / (instance_data + EPSILON)
    return res


if __name__ == "__main__":
    args = importlib.import_module(config.config_file)

    set_seed(args.seed, args.use_cuda)
    recmodel(args)


