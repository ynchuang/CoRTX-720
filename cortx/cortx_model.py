import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import drop_feature


class contrast_generator(nn.Module):
    def __init__(self, predict_model, column_data, mean_value_data, index_to_data, device='cuda:2'):
        super(contrast_generator, self).__init__()
        self.predict_model = predict_model
        self.column_data = column_data
        self.value = mean_value_data
        self.data_dict = index_to_data
        self.device = device
        self.drop_prob = 0.5

    def forward(self, model, data_idx, pos_num, neg_num):
        fea_num = len(self.value)
        targ_data = [ self.data_dict[int(idx)] for idx in data_idx ]

        pos_mask = np.random.choice([1, 0], size=(pos_num, fea_num))
        pos_subsample = self.pos_drop_sampling_mask(targ_data, pos_mask) 

        targ_tensor = torch.tensor(np.stack(targ_data)).to(self.device)
        pos_tensor = torch.tensor(np.stack(pos_subsample)).to(self.device)
        
        tar_exp = model(targ_tensor.float())
        pos_exp = model(pos_tensor.float())

        return tar_exp, pos_exp

    def pos_drop_sampling_mask(self, data, S):
        S = np.repeat(np.expand_dims(S, axis=0), repeats=len(data), axis=0)
        sample_num = S.shape[1]
        mask_value = torch.tensor(np.repeat(np.expand_dims([self.value]*sample_num, axis=0), repeats=len(data), axis=0))
        data = torch.tensor(data).repeat(1, sample_num, 1)
        pd_ref = pd.DataFrame(np.array(data)[:,0,:], columns=self.column_data)
        ref_model_input = {name: pd_ref[name] for idx, name in enumerate(self.column_data)}
        ref_score, _ = self.predict_model.predict(ref_model_input)

        train = drop_feature(data, self.drop_prob)
        inv_bool_mask = ~torch.gt(train, 0)
        reference_value = mask_value * inv_bool_mask.int()
        train = (train+reference_value).detach().cpu().numpy()

        for i in range(sample_num):
            pd_train = pd.DataFrame(train[:,i,:], columns=self.column_data)
            train_model_input = {name: pd_train[name] for idx, name in enumerate(self.column_data)}
            pred_score, _ = self.predict_model.predict(train_model_input)
            try:
                pos_gap = np.concatenate((pos_gap, pred_score), axis=1)
            except:
                pos_gap = pred_score
        score_gap = np.absolute(pos_gap - ref_score)
        rank_pos_list = np.argmin(score_gap, axis=1)
        best_pos = np.stack([ x[rank_pos_list[idx]] for idx, x in enumerate(train) ])
        return best_pos


class tab_mlp(nn.Module):
    def __init__(self, input_dim, output_dim, class_num=2, layer_num=1, hidden_dim=64, activation=None):
        super(tab_mlp, self).__init__()
        self.mlp = nn.ModuleList()
        self.layer_num = layer_num
        self.activation = eval(activation)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_str = activation
        self.class_num = class_num

        if layer_num == 1:
            layer1 = nn.Linear(input_dim, output_dim)
            self.mlp.append(layer1)

        else:
            for layer_index in range(layer_num):
                if layer_index == 0:
                    layer1 = nn.Linear(input_dim, hidden_dim)
                elif layer_index == layer_num - 1:
                    layer1 = nn.Linear(hidden_dim, output_dim)
                else:
                    layer1 = nn.Linear(hidden_dim, hidden_dim)
                self.mlp.append(layer1)

    def forward(self, x):
        for layer_index in range(self.layer_num - 1):

            layer = self.mlp[layer_index]
            if self.activation == None:
                x = layer(x)
            else:
                x = layer(x)
                x = self.activation(x)

        layer_lst = self.mlp[-1]
        y = layer_lst(x)

        return y

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)[:, 1]