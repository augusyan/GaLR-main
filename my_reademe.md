
#  文件对应改动
checkpoint_MultiGrained_demo：
- 对应my_engine_bert，bert权重不固定


rsitmd_GaLR_MIDF_MultiGrained_demo1：
- 对应my_engine_bert，bert权重全部固定


rsitmd_GaLR_MIDF_MultiGrained_demo2
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.00002
===================== Ave Score (3-fold verify) =================
r1i:0.07374631268436578
r5i:0.9587020648967551
r10i:1.8436578171091444
r1t:0.13274336283185842
r5t:0.9734513274336282
r10t:2.1386430678466075
mr:1.0201573254670602

rsitmd_GaLR_MIDF_MultiGrained_demo2_1
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.00005
r1i:0.0
r5i:1.0324483775811208
r10i:2.5073746312684366
r1t:0.2949852507374631
r5t:1.3126843657817109
r10t:2.256637168141593
mr:1.2340216322517208

rsitmd_GaLR_MIDF_MultiGrained_demo2_2
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.0002
r1i:0.14749262536873156
r5i:1.1799410029498525
r10i:2.2123893805309733
r1t:0.2507374631268437
r5t:1.1799410029498525
r10t:2.1533923303834808
mr:1.1873156342182891

rsitmd_GaLR_MIDF_MultiGrained_demo3
- 对应my_engine_bert，sentbert 

rsitmd_GaLR_MIDF_MultiGrained_demo4
- 对应my_engine_bert，adamw，lr=0.00002, cos metric,将val 和 test部分的bert相关数据集划分输入进行更正