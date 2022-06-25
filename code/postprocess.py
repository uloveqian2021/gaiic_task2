# -*- coding:utf-8 -*-
"""
@author : wang bq
@time   :22-4-22 下午7:24
@IDE    :PyCharm
@document   :postprocess.py
"""

from my_utils.util import get_entity_bio
maxlen = 128


def load_data(filename):
    """
    :param filename:  data path
    :return: [(text,[type, start_index, end_index),(......)]
    """
    data = {}
    i = 0
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            if line == "\n":
                if words:
                    labels = get_entity_bio(labels)
                    words = words[:maxlen]
                    # data.append((''.join(words), labels))
                    # data.append((words, labels))
                    data[i] = (''.join(words), labels)
                    words = []
                    labels = []
                    i += 1
            else:
                splits = line.split(" ")
                word = splits[0].replace("\n", "")
                if word == '':
                    word = ' '
                words.append(word)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:  # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            words = words[:maxlen]
            labels = get_entity_bio(labels)
            # data.append((''.join(words), labels))
            # data.append((words, labels))
            data[i] = (''.join(words), labels)

    return data


all_res = []

for i in range(10):
    rr = load_data(f'../data/public_data/bert-EfficientGlobalPointer-32-2e-05-pgd-kf{i}-unlabel.txt')
    # print(rr)
    print(i, rr[100])
    all_res.append(rr)


def predict(p_data, _submit_name):
    """评测函数
    """
    res = []
    fw = open(_submit_name, 'w', encoding='utf-8')
    for text, r in zip(p_data, res):
        labels = ['O'] * len(text[0])
        for t in r:
            labels[t[1]] = 'B-' + t[0]
            labels[t[1] + 1:t[2] + 1] = ['I-' + t[0]] * (t[2] - t[1])
        assert len(text[0]) == len(labels)
        for w, l in zip(text[0], labels):
            fw.write(w + ' ' + l + '\n')
        fw.write('\n')
    return ''


fw = open('../data/public_data/vote_unlabel_pseudo_tag420.txt', 'w', encoding='utf-8')
res_n = {}
for k, _ in all_res[0].items():
    res_k = {}
    num = 0
    text = list(_[0])
    for i in range(10):
        v = all_res[i][k][1]
        if v:
            num += 1
        for i in v:
            if tuple(i) not in res_k:
                res_k[tuple(i)] = 1
            else:
                res_k[tuple(i)] += 1
    res = sorted(res_k.items(), key=lambda x: x[1], reverse=True)
    res = [itm[0] for itm in res if itm[1]/num >= 0.5]
    print(text)
    print(res)
    print()
    labels = ['O'] * len(text)
    for t in res:
        labels[t[1]] = 'B-' + t[0]
        labels[t[1] + 1:t[2] + 1] = ['I-' + t[0]] * (t[2] - t[1])
    assert len(text) == len(labels)
    for w, l in zip(text, labels):
        fw.write(w + ' ' + l + '\n')
    fw.write('\n')
