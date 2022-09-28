# CoRTX-CE on feature ranking task
python3 -W ignore CoRTX_CE_adult.py --bs 1024 --exp_epoch 200 --temper 0.02 --pos_num 30 --neg_num 1024 --head_propor 0.25 --pretrain 1
echo 'Above Performance is on Feature Importance Ranking'
echo ''

# CoRTX-MSE on feature attribution task
python3 -W ignore CoRTX_MSE_adult.py --bs 1024 --exp_epoch 200 --temper 0.02 --pos_num 30 --neg_num 1024 --head_propor 0.25 --prot_reg 1e-5 --pretrain 1
echo 'Above Performance is on Feature Attribution'
