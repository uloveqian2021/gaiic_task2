# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : wangbingqian@boe.com.cn
@time   :22-4-20 上午9:26
@IDE    :PyCharm
@document   :trans_pseudo_data.py
"""
import random
all_data = []
with open('unlabeled_train_data.txt', 'r', encoding='utf-8') as tf:
    for line in tf.readlines():
        print(line)
        all_data.append(line.strip())

random.shuffle(all_data)
with open('../public_data/unlabeled_test_40000.txt', 'w', encoding='utf-8') as fw:
    for line in all_data[:40000]:
        for i in line:
            fw.write(i + '\n')
        fw.write('\n')
