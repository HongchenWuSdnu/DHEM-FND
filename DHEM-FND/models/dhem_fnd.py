import os

from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from tqdm import tqdm
import torch.nn as nn
from models.layers import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import math
import hdbscan
import numpy as np
from torch.nn.parameter import Parameter
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from umap.umap_ import UMAP
from scipy.stats import gaussian_kde
# ========== 工具函数 ==========
def cal_length(x):
    """计算L2范数"""
    return torch.norm(x, p=2, dim=1, keepdim=True)

def norm(x):
    """L2归一化"""
    return x / cal_length(x)

def convert_to_onehot(label, batch_size, num):
    """转换为one-hot编码"""
    return torch.zeros(batch_size, num).cuda().scatter_(1, label, 1)


class MemoryNetwork(nn.Module):
    """领域记忆网络，用于存储和更新领域特征"""

    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super().__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32  # 温度系数

        # 特征转换层
        self.topic_fc = nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = nn.Linear(input_dim, emb_dim, bias=False)

        # 初始化领域记忆库
        self.domain_memory = {
            i: torch.zeros(memory_num, emb_dim).cuda()
            for i in range(domain_num)
        }

    def forward(self, feature, category):
        """前向传播，计算领域注意力"""
        feature = norm(feature)

        # 获取领域记忆库
        domain_memory = [self.domain_memory[i] for i in range(self.domain_num)]

        # 计算各领域注意力
        sep_embeddings = [
            torch.mm(
                nn.functional.softmax(
                    torch.mm(self.topic_fc(feature), mem.T) * self.tau,
                    dim=1
                ),
                mem
            ).unsqueeze(1)
            for mem in domain_memory
        ]

        # 融合领域特征
        domain_att = torch.bmm(
            torch.cat(sep_embeddings, 1),
            self.domain_fc(feature).unsqueeze(2)
        ).squeeze()

        return nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

    def write(self, all_feature, category):
        """更新记忆库"""
        # 按领域分组特征
        fea_dict = defaultdict(list)
        for domain_idx, fea in zip(category.cpu().tolist(), all_feature):
            fea_dict[domain_idx].append(fea)  # domain_idx 已为 Python int 类型

        # 更新每个领域的记忆
        for domain, feas in fea_dict.items():
            feas = torch.stack(feas)

            # 计算注意力权重
            att = nn.functional.softmax(
                torch.mm(self.topic_fc(feas), self.domain_memory[domain].T) * self.tau,
                dim=1
            ).unsqueeze(2)

            # 生成新记忆并更新
            new_mem = (feas.unsqueeze(1).repeat(1, self.memory_num, 1) * att).mean(dim=0)
            att_mean = att.mean(dim=0)

            self.domain_memory[domain] = (
                    self.domain_memory[domain] * (1 - 0.05 * att_mean) +
                    0.05 * new_mem
            )


class DHEMFNDModel(nn.Module):
    """主模型"""

    def __init__(self, emb_dim, mlp_dims, dropout,
                 semantic_num, emotion_num, style_num,
                 LNN_dim, domain_num, dataset):
        super().__init__()
        # 基础配置
        self.dataset = dataset
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10

        # 专家数量配置
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.domain_num_expert = 3  # 政治、生活、娱乐
        self.LNN_dim = self.domain_num_expert

        # 特征维度配置
        self.fea_size = 256
        self.emb_dim = emb_dim

        # 预训练模型加载
        self.bert = {
            'ch': BertModel.from_pretrained('hfl/chinese-bert-wwm-ext',
                                   mirror='https://hf-mirror.com'),
            'en': RobertaModel.from_pretrained('roberta-base',
                                      mirror='https://hf-mirror.com')
        }[dataset].requires_grad_(False)

        # 特征提取器配置
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        # ===== 专家网络初始化 =====
        # 语义专家
        self.content_expert = nn.ModuleList([
            cnn_extractor(feature_kernel, emb_dim)
            for _ in range(semantic_num)
        ])

        # 情感专家
        input_dim = 47 * 5 if dataset == 'ch' else 38 * 5
        self.emotion_expert = nn.ModuleList([
            MLP(input_dim, [256, 320], dropout, False)
            for _ in range(emotion_num)
        ])

        # 风格专家
        style_dim = 48 if dataset == 'ch' else 32
        self.style_expert = nn.ModuleList([
            MLP(style_dim, [256, 320], dropout, False)
            for _ in range(style_num)
        ])


        #专家融合
        self.base_experts = nn.ModuleList([
            *self.content_expert,  # 语义专家
            *self.emotion_expert,  # 情感专家
            *self.style_expert  # 风格专家
        ])
        # 计算基础专家数量
        self.num_base_experts = len(self.content_expert) + len(self.emotion_expert) + len(self.style_expert)
        # 领域专家
        # self.domain_expert = nn.ModuleList([
        #     MLP(3, [256, 320], dropout, False)
        #     for _ in range(self.domain_num_expert)
        # ])
        domain_feature_dim = 3
        expert_input_dim = self.num_base_experts * 320 + domain_feature_dim
        self.domain_expert = nn.ModuleList([
            MLP(expert_input_dim, [512, 320], dropout, False)
            for _ in range(self.domain_num_expert)
        ])

        # ===== 门控机制 =====
        # self.gate = nn.Sequential(
        #     nn.Linear(emb_dim * 2, mlp_dims[-1]),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dims[-1], self.LNN_dim),
        #     nn.Softmax(dim=1)
        # )

        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.domain_num_expert),  # 仅对领域专家加权
            nn.Softmax(dim=1)
        )

        # ===== 记忆网络 =====
        mem_input_dim = emb_dim + (47 * 5 + 48 if dataset == 'ch' else 38 * 5 + 32)
        self.domain_memory = MemoryNetwork(
            mem_input_dim, mem_input_dim, domain_num, self.memory_num
        )

        # ===== 分类器 =====
        self.classifier = MLP(3, mlp_dims, dropout)
        self.domain_embedder = nn.Embedding(domain_num, emb_dim)
        self.attention = MaskAttention(emb_dim)
        self.all_feature = {}

        # 权重初始化
        self.weight = Parameter(torch.Tensor(1, self.LNN_dim, 320)).cuda()  # [1, 3, 320]
        nn.init.uniform_(
            self.weight,
            -1 / math.sqrt(self.LNN_dim),
            1 / math.sqrt(self.LNN_dim)
        )

    def forward(self, **kwargs):
        """统一后的前向传播逻辑"""
        # 提取输入特征
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        category = kwargs['category']
        domain_feature = kwargs['domain_feature']

        # 合并情感特征
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        # BERT编码内容特征
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        gate_input_feature, _ = self.attention(content_feature, content_masks)

        # 记忆网络处理融合特征
        memory_input = torch.cat([gate_input_feature, emotion_feature, style_feature], dim=-1)
        memory_att = self.domain_memory(memory_input, category)

        # 生成域嵌入
        domain_emb_all = self.domain_embedder(torch.arange(self.domain_num, device=content.device))
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)
        specific_embedding = self.domain_embedder(category.view(-1, 1)).squeeze(1)

        # 专家特征融合
        base_features = []
        # 语义专家
        for expert in self.content_expert:
            base_features .append(expert(content_feature))
        # 情感专家
        for expert in self.emotion_expert:
            base_features .append(expert(emotion_feature))
        # 风格专家
        for expert in self.style_expert:
            base_features .append(expert(style_feature))

        base_features  = torch.cat(base_features, dim=1)  # [batch, total_experts, 320]
        # 拼接原始领域特征
        domain_input = torch.cat([
            base_features ,
            kwargs['domain_feature'].float()  # 假设domain_feature是3维
        ], dim=1)

        # 领域专家
        domain_outputs = []
        for expert in self.domain_expert:
            out = expert(domain_input).unsqueeze(1)  # [batch, 320]
            domain_outputs.append(out)
        domain_outputs = torch.cat(domain_outputs, dim=1)

        # # 维度检查
        # print(f"基础专家数量: {self.num_base_experts}")
        # print(f"基础特征维度: {base_features .shape}")  # 应为 [batch_size, num_base_experts*320]
        # print(f"领域特征维度: {kwargs['domain_feature'].shape}")  # 应为 [batch_size, 3]
        # print(f"拼接后维度: {domain_input.shape}")  # 应为 [batch_size, num_base_experts*320 + 3]
        #
        # # MLP输入维度
        # first_linear = self.domain_expert[0].net[0]
        # print(f"领域专家第一层权重维度: {first_linear.weight.shape}")  # 应匹配 [num_base_experts*320 + 3, 512]

        # 特征变换与门控机制
        embed_log = torch.log1p(torch.abs(domain_outputs ) + 1e-7)
        # 维度对齐操作
        lnn_out = torch.matmul(
            self.weight,  # [1, 3, 3]
            embed_log.permute(0, 2, 1)  # [64, 3, 320]
        )# 输出: [64, 3, 320]
        lnn_feature = lnn_out.permute(0,2,1)
        # 验证专家总数
        # print("Total Experts (理论):", self.total_experts)
        # print("实际拼接的专家数:", shared_feature.size(1))  # 应等于 self.total_experts
        #
        # # 验证权重矩阵维度
        # print("self.weight 形状:", self.weight.shape)  # 应为 (1, LNN_dim, total_experts)
        #
        # # 验证 embed_x_log 维度
        # print("embed_x_log 形状:", embed_x_log.shape)  # 应为 (batch_size, total_experts, 320)

        # 门控融合
        gate_input = torch.cat([specific_embedding, general_domain_embedding], dim=-1)
        # ==== 门控融合 ====
        gate_value = self.gate(gate_input).view(-1, 1, self.domain_num_expert)
        output = torch.bmm(gate_value, lnn_feature).squeeze()
        # print(f"lnn_out shape: {lnn_out.shape}")  # 应为 [64, 3, 320]
        # print(f"gate_value shape: {gate_value.shape}")  # 应为 [64, 1, 3]
        # print(f"output shape: {output.shape}")  # 应为 [64, 320]
        return torch.sigmoid(self.classifier(output).squeeze(1))


    def save_feature(self, **kwargs):
        # 输入验证
        content = kwargs['content']
        assert content.dtype == torch.long, \
            f"输入ID类型错误！期望 torch.long，实际得到 {content.dtype}"

        # 关键参数打印（调试用）
        # print(f"""
        #     [输入验证]
        #     content 类型: {content.dtype} (应为 torch.int64)
        #     content_emotion 类型: {kwargs['content_emotion'].dtype} (应为 torch.float32)
        #     category 类型: {kwargs['category'].dtype} (应为 torch.int64)
        #     """)
        # 特征处理逻辑保持不变
        content_masks = kwargs['content_masks']
        comments = kwargs['comments']
        comments_masks = kwargs['comments_masks']
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        style_feature = kwargs['style_feature']
        category = kwargs['category']

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())


    def init_memory(self):
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            # 使用HDBSCAN进行聚类
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True).fit(all_feature)
            labels = clusterer.labels_
            unique_labels = set(labels)
            centers = []
            for label in unique_labels:
                if label != -1:  # 忽略噪声点
                    center = np.mean(all_feature[labels == label], axis=0)
                    centers.append(center)
            centers = np.array(centers)
            if len(centers) < self.memory_num:
                # 如果聚类中心数量不足，可以随机选择一些点作为补充
                random_indices = np.random.choice(len(all_feature), self.memory_num - len(centers), replace=False)
                additional_centers = all_feature[random_indices]
                centers = np.concatenate([centers, additional_centers])
            centers = torch.from_numpy(centers).cuda()
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)

    def extract_decoupled_features(self, ** kwargs):
        """提取解耦特征（语义、情感、风格）"""
        # 从输入中提取必要字段
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        category = kwargs['category']

        # ==== 内部生成必要特征 ====
        # 生成content_feature（通过BERT）
        with torch.no_grad():
            content_feature = self.bert(content, attention_mask=content_masks)[0]  # [batch, seq_len, emb_dim]

        # 合并情感特征
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        # ==== 特征提取 ====
        # 语义专家
        semantic_features = [expert(content_feature) for expert in self.content_expert]

        # 情感专家
        emotion_features = [expert(emotion_feature) for expert in self.emotion_expert]

        # 风格专家
        style_features = [expert(style_feature) for expert in self.style_expert]

        return (
            torch.stack(semantic_features, dim=1),  # [batch, sem_num, dim]
            torch.stack(emotion_features, dim=1),  # [batch, emo_num, dim]
            torch.stack(style_features, dim=1),  # [batch, sty_num, dim]
            category
        )


class Trainer:
    """模型训练器，封装训练和评估流程"""

    def __init__(
            self,
            emb_dim: int,  # 嵌入层维度
            mlp_dims: list,  # MLP层维度列表
            use_cuda: bool,  # 是否使用GPU
            lr: float,  # 学习率
            dropout: float,  # Dropout概率
            train_loader,  # 训练数据加载器
            val_loader,  # 验证数据加载器
            test_loader,  # 测试数据加载器
            category_dict: dict,  # 类别字典
            weight_decay: float,  # 权重衰减系数
            save_param_dir: str,  # 参数保存路径
            semantic_num: int,  # 语义专家数量
            emotion_num: int,  # 情感专家数量
            style_num: int,  # 风格专家数量
            lnn_dim: int,  # LNN维度
            dataset: str,  # 数据集名称
            early_stop: int = 5,  # 早停轮数
            epoches: int = 100  # 总训练轮数

    ):
        # 基础配置
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 模型参数
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.model = None
        self._init_model()
        self._init_components()
        # 初始化保存目录
        os.makedirs(save_param_dir, exist_ok=True)  # 自动创建目录
        self.save_param_dir = save_param_dir

    def train(self, logger=None):
        """训练主流程"""
        # 初始化日志
        if logger:
            logger.info('Start training...')

        # 模型初始化
        self._init_model()
        self._init_components()

        # 记忆网络初始化
        self._init_memory()
        print('Memory initialization completed')

        # 训练循环
        best_metric = None
        for epoch in range(self.epoches):
            # 单epoch训练
            avg_loss = self._train_epoch(epoch)
            #绘图
            # self.visualize_features(epoch=epoch)
            # 验证评估
            val_results = self.test(self.val_loader)

            # 早停与模型保存
            stop_flag = self._handle_early_stop(val_results, epoch)
            if stop_flag:
                break

        # 最终测试
        return self._final_test(logger)

    def test(self, dataloader):
        """评估流程"""
        self.model.eval()
        preds, labels, categories = [], [], []

        # 批量推理
        for batch in tqdm(dataloader, desc="Testing"):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                outputs = self.model(**batch_data)

                # 收集结果
                preds.extend(outputs.cpu().numpy().tolist())
                labels.extend(batch_data['label'].cpu().numpy().tolist())
                categories.extend(batch_data['category'].cpu().numpy().tolist())

        return metrics(labels, preds, categories, self.category_dict)

    # -------------------- 内部方法 --------------------
    def _init_model(self):
        """初始化模型组件"""
        self.model = DHEMFNDModel(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
            semantic_num=self.semantic_num,
            emotion_num=self.emotion_num,
            style_num=self.style_num,
            LNN_dim=self.lnn_dim,
            domain_num=len(self.category_dict),
            dataset=self.dataset
        )
        if self.use_cuda:
            self.model = self.model.cuda()

    def _init_components(self):
        """初始化训练组件"""
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.98
        )
        self.recorder = Recorder(self.early_stop)

    def _init_memory(self):
        """初始化记忆网络"""
        self.model.train()
        for batch in tqdm(self.train_loader, desc="Initializing memory"):
            batch_data = data2gpu(batch, self.use_cuda)
            self.model.save_feature(**batch_data)
        self.model.init_memory()

    def _train_epoch(self, epoch):
        """单epoch训练"""
        self.model.train()
        avg_loss = Averager()

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
            # 前向计算
            batch_data = data2gpu(batch, self.use_cuda)
            self.optimizer.zero_grad()
            outputs = self.model(**batch_data)

            # 损失计算
            loss = self.loss_fn(outputs, batch_data['label'].float())

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 更新记忆
            with torch.no_grad():
                self.model.write(**batch_data)

            # 记录损失
            avg_loss.add(loss.item())

        # 学习率调整
        self.scheduler.step()
        print(f'Epoch {epoch + 1} | Avg Loss: {avg_loss.item():.4f}')
        return avg_loss

    def _handle_early_stop(self, results, epoch):
        """处理早停逻辑"""
        mark = self.recorder.add(results)

        if mark == 'save':
            # 保存最佳模型
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_param_dir, 'DHEM-FND.pkl')
            )
            self.best_mem = self.model.domain_memory.domain_memory
            return False
        elif mark == 'esc':
            print(f'Early stopping at epoch {epoch + 1}')
            return True
        return False

    def _final_test(self, logger):
        """最终测试阶段"""
        # 加载最佳模型
        self.model.load_state_dict(
            torch.load(os.path.join(self.save_param_dir, 'DHEM-FND.pkl'))
        )
        self.model.domain_memory.domain_memory = self.best_mem

        # 执行测试
        test_results = self.test(self.test_loader)

        # 记录日志
        if logger:
            logger.info("Final test results:\n%s", test_results)
        print("Final Test Results:", test_results)

        return test_results, os.path.join(self.save_param_dir, 'DHEM-FND.pkl')

    def visualize_features(self, epoch=None):
        """增强版：添加HDBSCAN聚类边界与轮廓"""
        self.model.eval()

        # ===== 1. 类别配置（与图片完全一致） =====
        category_config = {
            0: {"name": "Technology", "color": "#1f77b4"},  # 蓝色
            1: {"name": "Military", "color": "#ff7f0e"},  # 橙色
            2: {"name": "Education", "color": "#2ca02c"}  # 绿色
        }

        # ===== 2. 数据过滤（仅保留三个有效类别） =====
        features, labels = [], []
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="特征提取"):
                batch_data = data2gpu(batch, self.use_cuda)
                sem_feat, emo_feat, sty_feat, categories = self.model.extract_decoupled_features(**batch_data)

                valid_mask = torch.isin(categories, torch.tensor([0, 1, 2], device=categories.device))
                valid_categories = categories[valid_mask]
                valid_feat = sem_feat.mean(dim=1)[valid_mask]

                features.append(valid_feat.cpu().numpy())
                labels.extend(valid_categories.cpu().numpy().tolist())

        # ===== 3. 降维（保持与图片相同的坐标范围） =====
        reducer = UMAP(
            n_neighbors=25,  # 大邻域保持全局结构
            min_dist=0.3,  # 宽松间距匹配原图分布
            metric='cosine',
            random_state=42,
            init='spectral'  # 确保坐标轴范围稳定
        )
        X_red = reducer.fit_transform(StandardScaler().fit_transform(np.concatenate(features)))

        # 强制坐标轴范围与图片一致
        X_red[:, 0] = np.clip(X_red[:, 0], -5, 15)
        X_red[:, 1] = np.clip(X_red[:, 1], -10, 10)

        # ===== 4. HDBSCAN聚类（每个类别内部） =====
        plt.figure(figsize=(20, 12))

        # 绘制密度等高线（与原图一致）
        kde = gaussian_kde(X_red.T, bw_method=0.25)
        xgrid = np.linspace(-5, 15, 200)
        ygrid = np.linspace(-10, 10, 200)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        plt.contourf(Xgrid, Ygrid, Z.reshape(Xgrid.shape), levels=8, alpha=0.15, cmap='Greys')

        # ===== 5. 逐类别聚类与可视化增强 =====
        for cat_id in category_config:
            mask = np.array(labels) == cat_id
            X_class = X_red[mask]

            # 动态HDBSCAN参数
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=30 if cat_id == 2 else 15,  # Education大簇用更大min_size
                cluster_selection_epsilon=0.8
            )
            cluster_labels = clusterer.fit_predict(X_class)

            # 绘制聚类边界（凸包）
            from scipy.spatial import ConvexHull
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

            for cluster in unique_clusters:
                cluster_points = X_class[cluster_labels == cluster]
                if len(cluster_points) < 5: continue

                # 绘制凸包边界
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                             color=category_config[cat_id]["color"],
                             alpha=0.3,
                             lw=1.5,
                             zorder=1)

                # 绘制椭圆标记簇区域
                cov = np.cov(cluster_points.T)
                lambda_, v = np.linalg.eigh(cov)
                lambda_ = np.sqrt(lambda_)
                ell = Ellipse(xy=np.mean(cluster_points, axis=0),
                              width=lambda_[0] * 3, height=lambda_[1] * 3,
                              angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                              edgecolor=category_config[cat_id]["color"],
                              facecolor='none',
                              linestyle='--',
                              alpha=0.4)
                plt.gca().add_patch(ell)

            # 绘制原始点（保持原图样式）
            plt.scatter(
                X_class[:, 0], X_class[:, 1],
                color=category_config[cat_id]["color"],
                alpha=0.85,
                s=60,
                edgecolors='w',
                linewidths=0.6,
                label=f"{category_config[cat_id]['name']} (n={sum(mask)})",
                zorder=2
            )

        # ===== 6. 坐标轴与图例（精确匹配原图） =====
        plt.title("Enhanced Category Clustering (UMAP Projection)", fontsize=18, pad=20)
        plt.xlabel("UMAP-1", fontsize=14)
        plt.ylabel("UMAP-2", fontsize=14)
        plt.xlim(-5, 15)
        plt.ylim(-10, 10)

        # 构建图例（过滤零样本类别）
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=f'{config["name"]} (n={np.sum(np.array(labels) == cat_id)})',
                       markerfacecolor=config["color"], markersize=10, markeredgewidth=0.6, markeredgecolor='white')
            for cat_id, config in category_config.items()
        ]
        plt.legend(handles=legend_elements,
                   title="Categories",
                   bbox_to_anchor=(1.18, 1),
                   fontsize=11,
                   title_fontsize=13,
                   framealpha=0.9)

        # 保存高清图像
        output_dir = r'D:\GitHubFile\DHEM-FND\enhanced_clusters'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'clustered_epoch{epoch}.png'),
                    bbox_inches='tight', dpi=350, facecolor='white')
        plt.close()



    def _init_model(self):
        self.model = DHEMFNDModel(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
            semantic_num=self.semantic_num,
            emotion_num=self.emotion_num,
            style_num=self.style_num,
            LNN_dim=self.lnn_dim,
            domain_num=len(self.category_dict),
            dataset=self.dataset
        )
        if self.use_cuda:
            self.model = self.model.cuda()