# cd ../..
CARD=0

CUDA_VISIBLE_DEVICES=$CARD python my_train.py --path_opt option/RSITMD_mca/RSITMD_GaLR_my.yaml

CUDA_VISIBLE_DEVICES=$CARD python my_test_ave.py --path_opt option/RSITMD_mca/RSITMD_GaLR_my.yaml