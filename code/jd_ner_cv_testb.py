import torch
from torch import nn
from transformers import BertModel, BartModel, BertTokenizerFast
from my_utils.models import GlobalPointer, EfficientGlobalPointer, MutiHeadSelection, Biaffine, TxMutihead
from my_utils.util import *
from my_utils.progressbar import ProgressBar
from my_utils.modeling_cpt import CPTModel
from my_utils.modeling_nezha import NeZhaModel
from my_utils.configuration_nezha import NeZhaConfig
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import datetime
import gc
TIME_NOW = datetime.datetime.now().strftime('%Y%m%d')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

# TODO
head_type = 'EfficientGlobalPointer'
maxlen = 128
bert_type = 'bert'
use_cat = False
use_fgm = False
use_pgd = True
adv_k = 3
online = True

base_path = '../data/pretrain_model/checkpoint-3250000/'

assert head_type in ['GlobalPointer', 'MutiHeadSelection', 'Biaffine', 'TxMutihead', 'EfficientGlobalPointer']
if head_type in ['MutiHeadSelection', 'Biaffine', 'TxMutihead']:
    batch_size = 16
    learning_rate = 1e-5
    abPosition = False
    rePosition = True
else:
    batch_size = 32
    learning_rate = 2e-5

if bert_type == 'cpt':
    bert_path = f'{base_path}cpt-base'
elif bert_type == 'macbertl':
    bert_path = f'{base_path}macbert_large'
elif bert_type == 'bart':
    bert_path = f'{base_path}bart-base-chinese'
elif bert_type == 'rbt3':
    bert_path = f'{base_path}chinese_rbt3_pytorch'
elif bert_type == 'roberta':
    bert_path = f'{base_path}chinese_roberta_wwm_ext_pytorch'
elif bert_type == 'robertal':
    bert_path = f'{base_path}chinese_roberta_wwm_large_ext_pytorch'
elif bert_type == 'nezha':
    bert_path = f'{base_path}nezha-cn-base'
else:
    bert_path = f'{base_path}'


config_path = f'{bert_path}/config.json'
checkpoint_path = f'{bert_path}/pytorch_model.bin'
dict_path = f'{bert_path}/vocab.txt'
data_path = 'data/public_data'
train_p = f'{data_path}/train.txt'
dev_p = f'{data_path}/dev.txt'
test_p = '../data/contest_data/preliminary_test_b/word_per_line_preliminary_B.txt'


# tokenizer = Tokenizer(dict_path, do_lower_case=True)
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

save_model_name = f'../data/model_data/{bert_type}-{head_type}-{batch_size}-{learning_rate}'
submit_name = f'../data/public_data/{bert_type}-{head_type}-{batch_size}-{learning_rate}'

if use_cat:
    save_model_name += '-cat'
    submit_name += '-cat'
if use_fgm:
    save_model_name += '-fgm'
    submit_name += '-fgm'
if use_pgd:
    save_model_name += '-pgd'
    submit_name += '-pgd'


if bert_type in ['macbert', 'macbertl', 'rbt3', 'roberta', 'robertal']:
    bert_type = 'bert'

print(f'Now Start traing {save_model_name}')


def load_data(filename):
    """
    :param filename:  data path
    :return: [(text,[type, start_index, end_index),(......)]
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            if line == "\n":
                if words:
                    labels = get_entity_bio(labels)
                    words = words[:maxlen]
                    data.append((''.join(words), labels))
                    # data.append((words, labels))
                    words = []
                    labels = []
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
            data.append((''.join(words), labels))
            # data.append((words, labels))
    return data


train_data = load_data(train_p)
val_data = load_data(dev_p)
test_data = load_data(test_p)
# pseudo_data = load_data(pseudo_p)


cat = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
       '21', '22', '23', '24', '25', '26', '28', '29', '30', '31',
       '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
       '42', '43', '44', '46', '47', '48', '49', '50', '51', '52', '53', '54']
c_size = len(cat)
c2id = {c: idx for idx, c in enumerate(cat)}
id2c = {idx: c for idx, c in enumerate(cat)}


def find_index(offset_mapping, index):
    for idx, internal in enumerate(offset_mapping[1:]):  # 第一个是 [CLS]
        if internal[0] <= index < internal[1]:
            return idx + 1
    return None


class DatasGenerator(DataGenerator):
    """数据生成器
    """

    def __init__(self, max_len, _data, _tokenizer):
        super(DatasGenerator, self).__init__(_data, batch_size)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __iter__(self, is_random=True):
        batch_token_ids, batch_mask_ids, batch_type_ids, batch_label = [], [], [], []

        for is_end, d in self.sample(is_random):
            label = np.zeros((c_size, self.max_len, self.max_len))
            enc_context = tokenizer(d[0],
                                    return_offsets_mapping=True,
                                    max_length=self.max_len,
                                    truncation=True,
                                    padding='max_length')
            for entity_info in d[1]:
                start, end = entity_info[1], entity_info[2]
                offset_mapping = enc_context['offset_mapping']
                start = find_index(offset_mapping, start)
                end = find_index(offset_mapping, end)
                if start and end and start < self.max_len and end < self.max_len:
                    label[c2id[entity_info[0]], start, end] = 1

            batch_token_ids.append(enc_context['input_ids'])
            batch_mask_ids.append(enc_context['attention_mask'])
            batch_type_ids.append(enc_context['token_type_ids'])
            batch_label.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:  # 输出batch
                batch_token_ids = torch.from_numpy(np.array(batch_token_ids))
                batch_mask_ids = torch.from_numpy(np.array(batch_mask_ids))
                batch_type_ids = torch.from_numpy(np.array(batch_type_ids))
                batch_label = torch.from_numpy(np.array(batch_label))
                yield [batch_token_ids, batch_mask_ids, batch_type_ids, batch_label]
                batch_token_ids, batch_mask_ids, batch_type_ids, batch_label = [], [], [], []


class Net(nn.Module):
    def __init__(self, model_path, head_type):
        super(Net, self).__init__()
        self.DU_DG = False
        self.hide_size = 768
        if bert_type == 'cpt':
            self.DU_DG = True
        if use_cat:
            self.hide_size *= 2

        if head_type == 'GlobalPointer':
            self.head = GlobalPointer(c_size, 64, self.hide_size)
        elif head_type == 'EfficientGlobalPointer':
            self.head = EfficientGlobalPointer(c_size, 64, self.hide_size)
        elif head_type == 'MutiHeadSelection':
            self.head = MutiHeadSelection(self.hide_size, c_size,
                                          abPosition=abPosition,
                                          rePosition=rePosition,
                                          maxlen=maxlen,
                                          max_relative=64)
        elif head_type == 'Biaffine':
            self.head = Biaffine(self.hide_size, c_size, Position=abPosition)
        elif head_type == 'TxMutihead':
            self.head = TxMutihead(self.hide_size, c_size,
                                   abPosition=abPosition,
                                   rePosition=rePosition,
                                   maxlen=maxlen,
                                   max_relative=64)
        if bert_type == 'bert':
            self.bert = BertModel.from_pretrained(model_path)
        elif bert_type == 'bart':  # TODO
            self.bart = BartModel.from_pretrained(model_path)
            self.bert = self.bart.get_encoder()
        elif bert_type == 'cpt':
            self.bert = CPTModel.from_pretrained(model_path)
        elif bert_type == 'nezha':
            config = NeZhaConfig.from_pretrained(model_path)
            self.bert = NeZhaModel.from_pretrained(model_path, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.DU_DG:
            x1 = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            if use_cat:
                x2 = torch.cat([x1.encoder_last_hidden_state,
                                x1["decoder_hidden_states"][-1]], dim=-1)
            else:
                x2 = x1.encoder_last_hidden_state

        elif bert_type == 'nezha':
            x2, x1 = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            # x2 = x1.last_hidden_state  # (batch_size, seq_len, hidden_size)
        else:
            x1 = self.bert(input_ids, attention_mask=attention_mask)
            x2 = x1.last_hidden_state  # (batch_size, seq_len, hidden_size)
        logits = self.head(x2, mask=attention_mask)
        return logits


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    # y_pred = (batch,length,length,labels)
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = torch.gt(y_pred, 0)
    count_acc = torch.sum(y_true * y_pred).item()
    count_all = torch.sum(y_true + y_pred).item()
    count_p = torch.sum(y_pred).item()
    count_t = torch.sum(y_true).item()
    return count_acc, count_all, count_p, count_t


def get_sample_precision(y_pred, y_true):
    y_pred = torch.gt(y_pred, 0).float()
    return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)


def train(data_loader, _model, loss_fn, _optimizer):
    _model.train()
    numerate, denominator, all_p, all_t = 0, 0, 0, 0
    for batch, data in enumerate(data_loader):
        _model.zero_grad()
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        token_type_ids = data[2].to(device)
        y = data[3].to(device)
        p = _model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(y, p)
        temp_n, temp_d, temp_p, temp_t = global_pointer_f1_score(y, p)
        all_p += temp_p
        all_t += temp_t
        numerate += temp_n
        denominator += temp_d
        # Back Propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=5.0, norm_type=2)

        if use_fgm:
            _model.zero_grad()
            fgm = FGM(_model)
            fgm.attack(emb_name='word_embeddings.')
            _p = _model(input_ids, attention_mask, token_type_ids)
            adv_loss = loss_fn(y, _p)
            adv_loss.backward()
            fgm.restore(emb_name='word_embeddings.')

        if use_pgd:
            _model.zero_grad()
            pgd = PGD2(_model, emb_name='word_embeddings.', epsilon=0.5, alpha=0.3)
            pgd.backup_grad()
            for t in range(adv_k):
                pgd.attack(is_first_attack=(t == 0))
                if t != adv_k - 1:
                    _model.zero_grad()
                else:
                    pgd.restore_grad()
                _p = _model(input_ids, attention_mask, token_type_ids)
                adv_loss = loss_fn(y, _p)
                adv_loss.backward()
            pgd.restore()

        _optimizer.step()
        _optimizer.zero_grad()
        pbar(batch, {'loss': loss.item(), 'f1': (2 * numerate / denominator)})
    print(f"Train  Acc: {(numerate / all_p):>4f}%; Recall: {(numerate / all_t):>4f}%; "
          f"F1: {(2 * numerate / denominator):>4f}%")


def test(data_loader, loss_fn, _model):
    size = len(data_loader)
    _model.eval()
    test_loss = 0
    numerate, denominator, all_p, all_t = 0, 0, 0, 0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            token_type_ids = data[2].to(device)
            y = data[3].to(device)
            p = _model(input_ids, attention_mask, token_type_ids)
            test_loss += loss_fn(y, p).item()
            temp_n, temp_d, temp_p, temp_t = global_pointer_f1_score(y, p)
            all_p += temp_p
            all_t += temp_t
            numerate += temp_n
            denominator += temp_d
    test_loss /= size * data_loader.batch_size
    test_f1 = 2 * numerate / denominator
    test_acc = numerate / (all_p+1e-10)
    test_recall = numerate / (all_t+1e-10)
    print(f"Test Result: Acc:{test_acc:>4f}%; Recall:{test_recall:>4f}%; "
          f"F1:{test_f1:>4f}%;Avg loss: {test_loss:>6f} \n")
    logger.info(f"Test Result: Acc:{test_acc:>4f}%; Recall:{test_recall:>4f}%; "
                f"F1:{test_f1:>4f}%;Avg loss: {test_loss:>6f} \n")

    return test_f1


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, _model, text, threshold=0):
        enc_context = tokenizer(text,
                                return_offsets_mapping=True,
                                max_length=256,
                                truncation=True)

        new_span, entities = [], []
        for i in enc_context['offset_mapping']:
            if i[0] == i[1]:  # CLS
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:  # ONE TOKEN
                    new_span.append([i[0]])
                else:  # MULTI TOKEN
                    new_span.append([i[0], i[-1] - 1])

        token_ids = torch.tensor(enc_context['input_ids']).long().unsqueeze(0).cuda()
        mask_ids = torch.tensor(enc_context['attention_mask']).long().unsqueeze(0).cuda()
        token_type_ids = torch.tensor(enc_context['token_type_ids']).long().unsqueeze(0).cuda()
        scores = _model(token_ids, mask_ids, token_type_ids)[0].data.cpu().numpy()
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (id2c[l], new_span[start][0], new_span[end][-1])
            )
        return entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    model.eval()
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(model, d[0]))
        T = set([tuple(i) for i in d[1]])
        print('R', R)
        print('T', T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict(p_data, _submit_name):
    """评测函数
    """
    res = []
    fw = open(_submit_name, 'w', encoding='utf-8')
    model.eval()
    for d in tqdm(p_data, ncols=100):
        R = set(NER.recognize(model, d[0]))
        res.append(R)
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


# model = Net(bert_path, head_type).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    logger = get_logger(save_model_name)
    logger.info(save_model_name)
    fold_nums = 10
    all_data = train_data + val_data
    # all_data = train_data + val_data + pseudo_data
    # print(train_data[:3])
    kf = KFold(n_splits=fold_nums, shuffle=True, random_state=520).split(all_data)
    for i, (train_fold, test_fold) in enumerate(kf):
        save_model_name_ = save_model_name + f'-kf{i}.pt'
        model = Net(bert_path, head_type).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        submit_name_ = submit_name + f'-kf{i}-test-b.txt'
        print("kFlod ", i, "/", fold_nums)
        # logger.info(f"Start kFlod {i}-{fold_nums}")
        # train_ = [all_data[i] for i in train_fold]
        # dev_ = [all_data[i] for i in test_fold]
        # train_generator = DatasGenerator(maxlen, train_data, tokenizer)
        # dev_generator = DatasGenerator(maxlen, val_data, tokenizer)
        # epochs = 15
        # max_F1 = 0
        # pbar = ProgressBar(n_total=len(train_generator),
        #                    desc='Training',
        #                    num_epochs=epochs)
        # for epoch in range(epochs):
        #     # print(f"Epoch {t + 1}\n-------------------------------")
        #     pbar.reset()
        #     pbar.epoch_start(current_epoch=epoch)
        #     train(train_generator, model, global_pointer_crossentropy, optimizer)
        #     F1 = test(dev_generator, global_pointer_crossentropy, model)
        #     if F1 > max_F1:
        #         max_F1 = F1
        #         torch.save(model.state_dict(), save_model_name_)
        #         print(f"Higher F1: {max_F1:>4f}%")
        # print("Model Train Done!")
        model.load_state_dict(torch.load(save_model_name_, map_location='cuda:0'))
        res = predict(test_data, submit_name_)
        gc.collect()

