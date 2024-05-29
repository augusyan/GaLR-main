# cd ../..
CARD=1

CUDA_VISIBLE_DEVICES=$CARD python my_train.py --path_opt option/RSITMD_mca/RSITMD_GaLR_sabert.yaml

# 记得打开
# CUDA_VISIBLE_DEVICES=$CARD python my_test_ave.py --path_opt option/RSITMD_mca/RSITMD_GaLR_sabert.yaml
