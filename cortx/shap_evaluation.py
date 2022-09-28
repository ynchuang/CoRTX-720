import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from random import sample
from tqdm import tqdm, trange
from scipy.stats import sem
from utils import save_checkpoint

@torch.no_grad()
def evaluation_mse(pred_model, model, head_model, test_loader, head_propor, pretrain, best_l2, column_data, mean_value_data, best_ste_l2):

    # initialize lists to monitor test loss and accuracy
    model.eval()
    head_model.eval()

    for data, target, value in test_loader:
        # Output for reg
        sample_num = data.shape[0]
        data_mean = torch.tensor(np.expand_dims([mean_value_data]*sample_num, axis=0)).squeeze()
        data0 = torch.zeros_like(data_mean)
        ref_model_input = {name: data0[:,idx].cpu().numpy() for idx, name in enumerate(column_data)}
        ref_score = pred_model.predict(ref_model_input)[0]

        ref_model_mean = {name: data_mean[:,idx].cpu().numpy() for idx, name in enumerate(column_data)}
        ref_score_mean= pred_model.predict(ref_model_mean)[0]

        train_model_input = {name: data[:,idx].cpu().numpy() for idx, name in enumerate(column_data)}
        pred_norm = pred_model.predict(train_model_input)[0]

        # Output for reg
        output_reg = head_model(model(data)).cpu()
        gap = torch.tensor(pred_norm) - torch.tensor(ref_score) - torch.sum(output_reg, 1, keepdim=True)
        additive_norm = gap / output_reg.shape[1]
        output_reg += additive_norm

        # Output for rank
        output_rank = head_model(model(data)).cpu()
        gap = torch.tensor(pred_norm) - torch.tensor(ref_score_mean) - torch.sum(output_rank, 1, keepdim=True)
        additive_norm = 1/output_rank.shape[1] * gap
        output_rank += additive_norm
        rank_output = torch.argsort(output_rank, dim=1)

        try:
            rank_shap_buf = torch.cat((rank_shap_buf, target), 0)
            rank_output_buf = torch.cat((rank_output_buf, rank_output), 0)
            value_output_buf = torch.cat((value_output_buf, output_reg), 0)
            value_shap_buf = torch.cat((value_shap_buf, value), 0)
        except:
            rank_shap_buf = target
            rank_output_buf = rank_output
            value_output_buf = output_reg
            value_shap_buf = value

    l2, ste_l2 = _l2_dist_attr(value_output_buf, value_shap_buf)
    if l2 <= best_l2 and not pretrain:
        best_l2 = l2
        best_ste_l2 = ste_l2
        save_checkpoint("./adult/weight/REG_model_adult_CoRTX_" + str(head_propor) + ".pth.tar",
                    best_l2 = best_l2,
                    pred_model = model,
                    mean_value = mean_value_data,
                    column_data = column_data,
                    head_linear_model = head_model,
                    )

    if pretrain:
        best_l2, best_ste_l2 = _l2_dist_attr(value_output_buf, value_shap_buf)
        return best_l2, best_ste_l2

    print("L2 Norm: %f | Std L2: %6f \n" %(float(l2), float(best_ste_l2)))
    return best_l2, best_ste_l2


@torch.no_grad()
def evaluation_ce(model, head_model, test_loader, head_propor, best_acc, best_std, pretrain):

    # initialize lists to monitor test loss and accuracy
    model.eval()
    head_model.eval()
    for data, target, value in test_loader:
        output = head_model(model(data))
        try:
            rank_shap_buf = torch.cat((rank_shap_buf, target), 0)
            value_output_buf = torch.cat((value_output_buf, output), 0)
            value_shap_buf = torch.cat((value_shap_buf, value), 0)
        except:
            rank_shap_buf = target
            value_output_buf = output
            value_shap_buf = value

    mAP, std_mAP = _acc_rank(value_output_buf, rank_shap_buf)

    if mAP >= best_acc and not pretrain:
        best_acc = mAP
        best_std = std_mAP
        save_checkpoint("./adult/weight/RANK_model_adult_CoRTX_" + str(head_propor) + ".pth.tar",
                    best_rank_acc = best_acc,
                    best_std_mAP = std_mAP,
                    pred_model = model,
                    head_linear_model = head_model,
                    )

    if pretrain:
        best_acc, best_std = _acc_rank(value_output_buf, rank_shap_buf)
        return best_acc, best_std

    print("Rank ACC: %f | Std Rank ACC: %6f \n" % (float(mAP), float(std_mAP)))
    return best_acc, best_std


def _acc_rank(rank_pred, rank_shap):
    feature_num = rank_shap.shape[1]
    rank_pred = rank_pred.cpu().reshape(-1, feature_num, feature_num).argmax(axis=2)
    rank_shap = rank_shap.cpu()
    mAP_weight = torch.tensor([[1./(feature_num-x) for x in range(feature_num)]])
    rank_mAP = torch.sum((rank_pred == rank_shap).type(torch.float)*mAP_weight, dim=1)/mAP_weight.sum()
    return torch.mean(rank_mAP), sem(rank_mAP)


def _l2_dist_attr(rank_pred, rank_shap):
    rank_pred = rank_pred.cpu().detach().numpy()
    rank_shap = rank_shap.cpu().detach().numpy()
    l2_loss = [ np.sum(np.power((np.array(rank_shap)[i]-np.array(rank_pred)[i]),2)) for i in range(len(rank_pred)) ]
    return np.mean(l2_loss), sem(l2_loss)
