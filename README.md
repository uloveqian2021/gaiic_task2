# 代码说明
uloveqian团队提交代码说明


## 环境配置
默认是python3.7


## 数据
仅仅使用了官方的竞赛数据集，按照9:1划分train:dev,存放于data/public_data 目录下，
其中train.json和dev.json文件只是方便读取文件转换了格式而已

## 预训练模型
使用了macbert训练模型，可以通过https://huggingface.co/hfl/chinese-macbert-base链接获得，
对code/run_pretrain.py中的BERT网络进行初始化

## 算法

### 整体思路介绍（必选）
使用了　EfficientGlobalPointer 进行实体标注，　参考https://kexue.fm/archives/8373


### 网络结构（必选）
使用BERT对文本进行编码，　使用EfficientGlobalPointer进行解码

### 损失函数（必选）
使用了　multilabel_categorical_crossentropy　参考https://kexue.fm/archives/7359

### 模型集成（可选）
使用了10折交叉验证，采用投票的方式生成了为标签，其中无标注数据随机抽取了4W条，testa 抽取了7000条, testb抽取了5000条


## 训练流程
对train.sh每一步进行描述，或者在train.sh中对每一步添加注释
python code/run_pretrain.py      采用无标签数据＋训练集＋testa 基于macbert继续训练了100 epoch
python code/jd_ner_cv.py 　　     基于上一步训练的模型进行１０折交叉验证生成无标注数据上的伪标签
python code/postprocess.py        利用投票的方式进行融合
python code/jd_ner_cv_testa.py 　　生成testa数据上的伪标签
python code/postprocess_testa.py
python code/jd_ner_cv_testb.py       生成testb数据上的伪标签
python code/postprocess_testb.py
python code/jd_ner.py            使用三个为标签数据加上训练集数据进行模型训练，并生成最终的结果
## 测试流程
对test.sh每一步进行描述，或者在test.sh中对每一步添加注释
python code/jd_ner.py           加载最好的模型训练，并生成最终的结果

## 其他注意事项
按照9:1划分train(3.6W):dev(0.4W),存放于data/public_data 目录下