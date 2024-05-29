
#  文件对应改动


### checkpoint_MultiGrained_demo：
- 对应my_engine_bert，bert权重不固定


### rsitmd_GaLR_MIDF_MultiGrained_demo1：
- 对应my_engine_bert，bert权重全部固定


### rsitmd_GaLR_MIDF_MultiGrained_demo2
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.00002
===================== Ave Score (3-fold verify) =================
r1i:0.07374631268436578
r5i:0.9587020648967551
r10i:1.8436578171091444
r1t:0.13274336283185842
r5t:0.9734513274336282
r10t:2.1386430678466075
mr:1.0201573254670602

### rsitmd_GaLR_MIDF_MultiGrained_demo2_1
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.00005
r1i:0.0
r5i:1.0324483775811208
r10i:2.5073746312684366
r1t:0.2949852507374631
r5t:1.3126843657817109
r10t:2.256637168141593
mr:1.2340216322517208

### rsitmd_GaLR_MIDF_MultiGrained_demo2_2
- 对应my_engine_bert，bert权重固定被no grad替代，替换度量位 dot product,lr=0.0002
r1i:0.14749262536873156
r5i:1.1799410029498525
r10i:2.2123893805309733
r1t:0.2507374631268437
r5t:1.1799410029498525
r10t:2.1533923303834808
mr:1.1873156342182891


## 更正了一些错误，将val 和 test部分的bert相关数据集划分输入进行更正

### rsitmd_GaLR_MIDF_MultiGrained_demo4
- 对应my_engine_bert，adamw，lr=0.00002, cos metric, epoch=20.将val 和 test部分的bert相关数据集划分输入进行更正
r1i:4.424778761061947
r5i:13.7905604719764
r10i:23.230088495575217
r1t:2.8761061946902657
r5t:12.448377581120944
r10t:22.52212389380531
mr:13.215339233038348

### rsitmd_GaLR_MIDF_MultiGrained_demo4_1
- lr=0.0002
r1i:1.4749262536873156
r5i:8.112094395280236
r10i:13.274336283185841
r1t:1.047197640117994
r5t:4.941002949852507
r10t:10.23598820058997
mr:6.51425762045231

### rsitmd_GaLR_MIDF_MultiGrained_demo4_2
- epoch=40

### rsitmd_GaLR_MIDF_MultiGrained_demo4_3
- adam   lr=0.0002
Nan

### rsitmd_GaLR_MIDF_MultiGrained_demo3
- 对应my_engine_bert，sentbert, adamw，lr=0.00002, cos metric, epoch=20 
r1i:0.2949852507374631
r5i:1.3274336283185841
r10i:2.7286135693215336
r1t:0.20648967551622419
r5t:1.710914454277286
r10t:3.3480825958702063
mr:1.6027531956735495

## 沿用原本的GALR操作
### 加入 nltk提取名字特征

