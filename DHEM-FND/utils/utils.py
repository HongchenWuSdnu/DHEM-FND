from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
            self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}

    # 初始化每个类别的结果存储
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    # 按类别分组
    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    # 计算总体指标
    if len(y_true) == 0:
        return {"error": "空数据集"}

    y_pred_round = np.round(y_pred)
    overall_metrics = {
        'metric': f1_score(y_true, y_pred_round, average='macro').round(4),
        'auc': roc_auc_score(y_true, y_pred).round(4) if len(np.unique(y_true)) >= 2 else None,
        'recall': recall_score(y_true, y_pred_round, average='macro').round(4),
        'precision': precision_score(y_true, y_pred_round, average='macro').round(4),
        'acc': accuracy_score(y_true, y_pred_round).round(4)
    }


    return overall_metrics

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'comments': batch[2].cuda(),
            'comments_masks': batch[3].cuda(),
            'content_emotion': batch[4].cuda(),
            'comments_emotion': batch[5].cuda(),
            'emotion_gap': batch[6].cuda(),
            'style_feature': batch[7].cuda(),
            'label': batch[8].cuda(),
            'category': batch[9].cuda(),
            'domain_feature': batch[10].cuda()
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'comments': batch[2],
            'comments_masks': batch[3],
            'content_emotion': batch[4],
            'comments_emotion': batch[5],
            'emotion_gap': batch[6],
            'style_feature': batch[7],
            'label': batch[8],
            'category': batch[9],
            'domain_feature': batch[10]
            }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v