import numpy as np
import argparse
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from random import sample
from tqdm import trange
from scipy.stats import sem
from utils import drop_feature, setup_seed
from cortx_model import tab_mlp, contrast_generator
from shap_evaluation import evaluation_ce

from contrastive.contrastive_model import DualBranchContrast
import contrastive.infonce as L


class ProtocalDatagenerator(Dataset):
    def __init__(self, x_filename, y_filename, head_propor, device='cuda'):
        self.head_propor = head_propor
        self.data = self.read_pd_file(x_filename)
        self.label = self.read_tesnor_file(y_filename)
        self.data = self.data.to(device)
        self.label = self.label.to(device)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def read_pd_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        data = data.drop(columns=["fnlwgt", "class", "label"])
        propor_train_len = int(len(data) * float(self.head_propor))
        propor_data = data[:propor_train_len]
        return torch.tensor(np.array(propor_data)).type(torch.float).detach().cpu()

    def read_tesnor_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        propor_train_len = int(len(data) * float(self.head_propor))
        propor_data = data[:propor_train_len]
        return torch.tensor(np.array(propor_data)).type(torch.float).detach().cpu()


class Datagenerator(Dataset):
    def __init__(self, x_filename_dict, device='cuda'):
        self.device = device
        self.data_emb_dict = x_filename_dict
        self.data_emb_list = list(x_filename_dict.values())
        self.data_idx_list = torch.tensor(list(x_filename_dict.keys())).to(device)

    def __getitem__(self, index):
        return self.data_idx_list[index], index
       
    def __len__(self):
        return len(self.data_emb_list)


class TestDatagenerator(Dataset):
    def __init__(self, x_filename, y_rank_filename, y_value_filename, device='cuda'):
        self.data = self.read_pd_file(x_filename)
        self.label = self.read_tesnor_file(y_rank_filename)
        self.value = self.read_tesnor_file(y_value_filename)
        self.data = self.data.to(device)
        self.label = self.label.to(device)
        self.value = self.value.to(device)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.value[index]

    def __len__(self):
        return len(self.data)

    def read_pd_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        data = data.drop(columns=["fnlwgt","class", "label"])
        return torch.tensor(np.array(data)).type(torch.float).detach().cpu()

    def read_tesnor_file(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return torch.tensor(np.array(data)).type(torch.float).detach().cpu()


def main(args):
    ###################
    # Device Setting
    ###################
    setup_seed(7)
    best_acc = 0
    best_std = 0
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:2'


    ###################
    # Load predictive model
    ###################
    model_checkpoint_fname = "./adult/adult_autoint_model.pth"
    predict_model = torch.load(model_checkpoint_fname)


    ###################
    # Load training data
    ###################
    print("Loading Data")
    index_to_data = defaultdict(list)
    mean_value_data = []
    column_data = []
    with open('./adult/adult_autoint_train.pickle', 'rb') as f:
        train_contras = pickle.load(f)
        train_contras = train_contras.drop(columns=['fnlwgt'])
    for col_name in train_contras.columns[:-2]:
        mean_value_data.append(train_contras[str(col_name)].mean())
        column_data.append(str(col_name))
    for idx, (didx, data) in enumerate(train_contras.iterrows()):
        index_to_data[idx].append(data.values.tolist()[:-2])


    ###################
    # Create dataloader
    ###################
    train_data = Datagenerator(index_to_data, device=device)
    test_data = TestDatagenerator(x_filename='./adult/adult_autoint_test.pickle',
                                  y_rank_filename='./adult/adult_autoint_test_rank.pickle',
                                  y_value_filename='./adult/adult_autoint_test_value.pickle',
                                  device=device)
    protocal_train_data = ProtocalDatagenerator(x_filename='./adult/adult_autoint_train.pickle',
                                                y_filename='./adult/adult_autoint_train_rank_all.pickle',
                                                head_propor=args.head_propor,
                                                device=device)

    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=512, shuffle=True)                                               
    protocal_train_loader = DataLoader(protocal_train_data, batch_size=256, shuffle=True)


    ################
    # Explainer setting
    ################
    print("Starting Building Model")
    hidden_unit = [256, 256, 256]

    pos_num = args.pos_num
    neg_num = args.neg_num
    temper = args.temper
    n_epochs = args.exp_epoch
    
    model = tab_mlp(input_dim=len(column_data), output_dim=256,
                    class_num=len(column_data), layer_num=3, hidden_dim=256,
                    activation="torch.nn.functional.elu")
    print(model)
    model.to(device)

    contrast_gen = contrast_generator(predict_model, column_data,
                                      mean_value_data, index_to_data,
                                      device)
    ccontras_loss = DualBranchContrast(loss=L.InfoNCE(tau=temper), mode='G2G')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-10)


    ################
    # Explanation Encoder Training
    ################
    model.train()
    for epoch in trange(1, n_epochs, desc="Explanation Encoder Training", unit="epochs"):
        
        # pretrain loading
        if args.pretrain:
            print(" ===== Start to use prtrain =====")
            checkpoint = torch.load('./adult/weight/model_adult_autoint_0.01.pth.tar')
            model = checkpoint["pred_model"]
            protocal_model = checkpoint["head_linear_model"]
            best_acc, std_mAP = evaluation_ce(model, protocal_model, test_loader, args.head_propor, best_acc, best_std, args.pretrain)
            break

	    # init training loss
        train_loss = 0.0
	    # train the model
        for data_idx, _ in train_loader:
            optimizer.zero_grad()
            tar, pos = contrast_gen(model, data_idx, pos_num, neg_num)
            loss = ccontras_loss(g1=pos, g2=tar.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

	    # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)


        ################
        # Explanation Head Training
        ################
        if epoch%5==0:

            print('----- Epoch: {} ----- \t Training Loss: {:.6f}'.format(epoch, train_loss))
            model.eval()

            print("Starting Building Explanation Head")
            protocal_n_epochs = args.prot_epoch
            protocal_model = tab_mlp(input_dim=hidden_unit[-1],
                                 output_dim=len(column_data) * len(column_data),
                                 layer_num=3, hidden_dim=256,
                                 activation="torch.nn.functional.elu")
            protocal_model.to(device)
            protocal_criterion = nn.CrossEntropyLoss().to(device)
            protocal_optimizer = torch.optim.Adam(protocal_model.parameters(), lr=5e-3)
            
            protocal_model.train()
            print("propor Length: ", len(protocal_train_loader.dataset))
            for epoch in trange(protocal_n_epochs, desc="Explanation Head Training", unit="epochs"):
                # init training loss
                train_loss = 0.0

                # train the model
                for data, ranking_gt in protocal_train_loader:
                    
                    protocal_optimizer.zero_grad()   
                    output = protocal_model(model(data))          
                    ranking_pred = output.reshape(-1, ranking_gt.shape[1])
                    ranking_target = ranking_gt.long().reshape(-1)

                    protocal_loss = protocal_criterion(ranking_pred, ranking_target)
                    protocal_loss.backward()
                    protocal_optimizer.step()
                    train_loss += protocal_loss.item()

                # calculate average loss over an epoch
                train_loss = train_loss/len(protocal_train_loader.dataset)
                # if epoch%100 ==0:
                #    print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

            best_acc, best_std = evaluation_ce(model, protocal_model, test_loader, args.head_propor, best_acc, best_std, args.pretrain)
    print("Best Rank ACC: %9f" % (float(best_acc)))
    print("Ste Rank Acc: %9f" %(float(best_std)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-Supervised L2E')

    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--exp_epoch", type=int, default=200)
    parser.add_argument("--prot_epoch", type=int, default=200)
    parser.add_argument("--pos_num", type=int, default=1)
    parser.add_argument("--neg_num", type=int, default=10)
    parser.add_argument("--temper", type=float, default=0.02)
    parser.add_argument("--exp_reg", type=float, default=1e-10)
    parser.add_argument("--prot_reg", type=float, default=1e-12)
    parser.add_argument('--head_propor', type=float, default=0.01)


    args = parser.parse_args()
    main(args)
