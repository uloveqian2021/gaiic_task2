# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : wangbingqian@boe.com.cn
@time   :22-3-22 下午6:04
@IDE    :PyCharm
@document   :split_data.py
"""
import json


def get_entity_bio2(label):
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(label):
        if tag.startswith("B-"):
            if chunk[2] != -1:                 # 前一个实体刚结束,需要保存
                chunks.append(chunk)
            chunk = [tag.split('-')[1], i, i]  # 当前实体的类型和初始位置
            if i == len(label) - 1:            # 单字情况且在句尾，需要保存
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            # 实体中间元素
            _type = tag.split('-')[1]
            if _type == chunk[0]:              # 这里是进行校验？
                chunk[2] = i
            if i == len(label) - 1:
                chunks.append(chunk)
        else:  # 非实体且上一个元素是实体，则保存
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]      # 元组重新初始化
    return chunks


def _read_text(input_file):
    lines = []
    with open(input_file, 'r') as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"words": words, "labels": labels})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                word = splits[0]
                if word == '':
                    word = ' '
                words.append(word)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"words": words, "labels": labels})
    return lines


f = open('./dev.json', 'w', encoding='utf-8')
i = 0
with open('dev.txt', 'r', encoding='utf-8') as tf:
    for line in tf.readlines():
        print(line)
        words, label = line.strip().split('\t')
        words = ''.join(words.split('\002'))
        label = label.split('\002')
        lb = get_entity_bio2(label)

        i += 1
        # print(words[lb[1][1]:lb[1][2]+1], lb[1])

        s = json.dumps({
            'text': words,
            'labels': label,
            'span_labels': lb,
        },
            ensure_ascii=False)
        f.write(s + '\n')

