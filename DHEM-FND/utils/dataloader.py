import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer

# 环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def _init_fn(worker_id):
    """DataLoader worker初始化函数"""
    np.random.seed(2021)


def read_pkl(path):
    """安全读取pickle文件"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"读取pickle文件失败: {path}") from e


def df_filter(df_data, category_dict):
    """过滤有效类别数据"""
    valid_categories = set(category_dict.keys())
    filtered_df = df_data[df_data['category'].isin(valid_categories)]

    # 检查过滤后数据是否为空
    if filtered_df.empty:
        raise ValueError(f"数据过滤后为空，请检查category_dict与数据的匹配性。有效类别: {valid_categories}")

    return filtered_df


def get_tokenizer(dataset_type):
    """获取适合数据集的tokenizer"""
    tokenizers = {
        'ch': ('hfl/chinese-bert-wwm-ext', BertTokenizer),
        'en': ('roberta-base', RobertaTokenizer)
    }
    model_name, tokenizer_class = tokenizers.get(dataset_type, (None, None))
    if not model_name:
        raise ValueError(f"不支持的数据集类型: {dataset_type}，可选: ch/en")
    return tokenizer_class.from_pretrained(model_name)


def text_to_tensors(texts, max_len, dataset_type):
    """文本序列化处理"""
    tokenizer = get_tokenizer(dataset_type)
    encoded = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    return encoded['input_ids'], encoded['attention_mask']


def safe_convert(x, feature_name):
    """安全转换特征数据"""
    try:
        # 处理可能的字符串表示（如 "[0.1, 0.2]"）
        if isinstance(x, str):
            x = eval(x)  # 注意安全性，确保数据来源可信

        arr = np.array(x, dtype=np.float32).flatten()
        return arr
    except Exception as e:
        raise ValueError(
            f"无法转换特征 {feature_name} 的值: {repr(x)}\n"
            f"错误类型: {type(x)}, 错误详情: {str(e)}"
        )


def word2input(texts, max_len, dataset):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext',
                                   mirror='https://hf-mirror.com') if dataset == 'ch' \
        else RobertaTokenizer.from_pretrained('roberta-base',mirror='https://hf-mirror.com')

    tokenizer_output = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    token_ids = tokenizer_output['input_ids'].long()  # 强制类型转换
    masks = tokenizer_output['attention_mask'].bool()

    return token_ids, masks

def process_emotion_data(df):
    """处理多维情感特征数据"""
    df = df.copy()

    # 生成领域特征（示例：根据category生成）
    df['domain_feature'] = df['category'].apply(
        lambda x: [1, 0, 0] if x == '政治' else [0, 1, 0] if x == '经济' else [0, 0, 1]
    )

    def convert_feature(series, feature_name):
        """带验证的特征转换"""
        # 转换每个元素
        converted = series.apply(
            lambda x: safe_convert(x, feature_name)
        )

        # 检查维度一致性
        lengths = converted.apply(len)
        if lengths.nunique() != 1:
            error_samples = series[lengths != lengths.mode()[0]].head(2)
            raise ValueError(
                f"特征 {feature_name} 存在不一致的维度\n"
                f"多数样本长度: {lengths.mode()[0]}\n"
                f"异常样本:\n{error_samples}"
            )
        return converted
        # 处理各特征列


    features = ['content_emotion', 'comments_emotion', 'emotion_gap']
    for feat in features:
        df[feat] = convert_feature(df[feat], feat)

    return df

class BertDataLoader:
    """BERT数据加载处理器"""

    def __init__(self, max_len, batch_size, category_dict, dataset_type, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset_type = dataset_type


    def _validate_params(self):
        """参数校验"""
        if not isinstance(self.category_dict, dict):
            raise TypeError("category_dict应为字典类型")
        if self.dataset_type not in ['ch', 'en']:
            raise ValueError("dataset_type应为'ch'或'en'")

    def load_data(self, data_path, shuffle=False, dataset_name=''):
        """加载并处理数据"""
        # 数据加载与预处理
        raw_df = read_pkl(data_path)
        filtered_df = df_filter(raw_df, self.category_dict)
        processed_df = process_emotion_data(filtered_df)

        # 特征提取
        content_ids, content_masks = self._process_text_features(processed_df['content'])
        comments_ids, comments_masks = self._process_text_features(processed_df['comments'])

        # 构建数据集
        dataset = self._build_dataset(processed_df, content_ids, content_masks,
                                      comments_ids, comments_masks)

        # 打印数据分布
        print(f"{dataset_name}数据集类别分布:\n{processed_df['category'].value_counts()}")

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )

    def _process_text_features(self, texts):
        """处理文本特征"""
        return text_to_tensors(texts.tolist(), self.max_len, self.dataset_type)

    def _build_dataset(self, df, content_ids, content_masks, comments_ids, comments_masks):
        """确保张量顺序与模型输入参数严格一致"""
        # 文本特征（必须在前）
        content_ids = content_ids.long()  # [batch, seq_len]
        content_masks = content_masks.bool()  # [batch, seq_len]

        # 评论特征
        comments_ids = comments_ids.long()  # [batch, seq_len]
        comments_masks = comments_masks.bool()  # [batch, seq_len]

        # 情感特征
        content_emotion = torch.tensor(np.stack(df['content_emotion']), dtype=torch.float32)
        comments_emotion = torch.tensor(np.stack(df['comments_emotion']), dtype=torch.float32)
        emotion_gap = torch.tensor(np.stack(df['emotion_gap']), dtype=torch.float32)

        # 风格特征
        style_feature = torch.tensor(np.vstack(df['style_feature'].apply(np.array)), dtype=torch.float32)

        # 标签和类别
        label = torch.tensor(df['label'].astype(int).values, dtype=torch.long)
        category = torch.tensor(df['category'].map(self.category_dict).values, dtype=torch.long)
        num_domains = 3
        domain_feature = torch.nn.functional.one_hot(category, num_classes=num_domains).float()
        return TensorDataset(
            # 输入特征（必须与模型参数顺序一致）
            content_ids,  # kwargs['content']
            content_masks,  # kwargs['content_masks']
            comments_ids,  # kwargs['comments']
            comments_masks,  # kwargs['comments_masks']

            # 其他特征
            content_emotion,  # kwargs['content_emotion']
            comments_emotion,  # kwargs['comments_emotion']
            emotion_gap,  # kwargs['emotion_gap']
            style_feature,  # kwargs['style_feature']

            # 标签和类别
            label,  # kwargs['label']
            category,  # kwargs['category']
            domain_feature
        )