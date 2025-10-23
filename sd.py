##屏蔽进度条，github中不支持显示，整个notebook都不显示了
import os
# 设置这个环境变量来禁用tqdm进度条
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import datasets
datasets.disable_progress_bar()

import os
import urllib.request
from pathlib import Path

# === 1. 全局配置 ===
URLPrefix = "https://pro-5gu0t2os8cdd45f2-1251420592.tcloudbaseapp.com/toxic-comment-classification"
data_dir = 'toxic-comment'
DATA_DIR = Path(data_dir)
FILENAMES = ["train.csv","test.csv","test_labels.csv","sample_submission.csv"]

BATCH_SIZE = 8
RANDOM_STATE = 123
NUM_WORKERS = 2


# === 2. 数据准备 ===
def prepare_csv_list():
    # 如果toxic-comment 不存在，创建该目录
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fileName in FILENAMES:
        URL = f"{URLPrefix}/{fileName}"
        DATA_FILE =DATA_DIR/fileName
        if not DATA_FILE.exists():
            print(f"⬇️ Downloading {fileName}...")
            with urllib.request.urlopen(URL) as r, open(DATA_FILE, "wb") as f:
                f.write(r.read())
        else:
            print(f"✅ already exists: {fileName} ")

import re
import pandas as pd
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset


class TextPreprocessor:
    """使用 GPT2Tokenizer 的文本预处理器"""

    def __init__(self,
                 model_name="gpt2",
                 max_seq_length=128,
                 add_special_tokens=True,
                 padding=True,
                 truncation=True):
        """
        Args:
            model_name (str): 使用的预训练 GPT2 分词器名称（如 "gpt2", "gpt2-medium"）
            max_seq_length (int): 最大序列长度
            add_special_tokens (bool): 是否添加特殊标记（如 BOS/EOS）
            padding (bool): 是否自动填充
            truncation (bool): 是否截断超长文本
        """
        self.max_seq_length = max_seq_length
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.truncation = truncation

        # ✅ 初始化 GPT2 分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 默认没有 pad_token，需要手动设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ GPT2Tokenizer 已加载: {model_name}")
        print(f"词表大小: {len(self.tokenizer)}")

    def clean_text(self, text: str) -> str:
        """可选的文本清理（保留基础清洗逻辑）"""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def build_vocab(self, texts):
        """兼容旧接口（GPT2Tokenizer 自带词汇表，无需手动构建）"""
        print("⚙️ 使用 GPT2Tokenizer 自带词汇表，无需手动构建。")
        return list(self.tokenizer.get_vocab().keys())

    def text_to_sequence(self, text):
        """文本转为 GPT-2 Token ID 序列"""
        cleaned_text = self.clean_text(text)
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_seq_length,
            padding='max_length' if self.padding else False,
            truncation=self.truncation,
            return_tensors=None
        )
        return encoding["input_ids"]

    def batch_encode(self, texts):
        """批量文本编码"""
        cleaned_texts = [self.clean_text(t) for t in texts]
        encodings = self.tokenizer(
            cleaned_texts,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_seq_length,
            padding='max_length' if self.padding else False,
            truncation=self.truncation,
            return_tensors=None
        )
        return encodings["input_ids"]


class ToxicCommentDataset(Dataset):
    """有毒评论数据集"""

    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        # self.labels = labels if labels is not None else [[0]*6]*len(texts)
        self.labels = labels if labels is not None else [[0]*6 for _ in range(len(texts))]

        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sequence = self.preprocessor.text_to_sequence(text)

        if not sequence or len(sequence) == 0:
            sequence = [self.preprocessor.tokenizer.pad_token_id] * self.preprocessor.max_seq_length


        # 创建attention mask（非零位置为1）
        attention_mask = [1 if token != 0 else 0 for token in sequence]

        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


def read_toxic_comments_real(data_dir, max_samples=None, is_train=True):
    """
    读取真实的Kaggle Toxic Comment Classification数据
    返回格式: (texts, labels, ids)
    """
    if is_train:
        csv_path = os.path.join(data_dir, 'train.csv')
        print(f"读取训练数据: {csv_path}")

        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)

        texts = df['comment_text'].tolist()
        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[label_columns].values.tolist()
        ids = df['id'].tolist()

        print(f"加载训练数据: {len(texts)} 条")
        print(f"标签分布: {dict(zip(label_columns, df[label_columns].sum().tolist()))}")

        return texts, labels, ids
    else:
        csv_path = os.path.join(data_dir, 'test.csv')
        print(f"读取测试数据: {csv_path}")

        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)

        texts = df['comment_text'].tolist()
        ids = df['id'].tolist()

        print(f"加载测试数据: {len(texts)} 条")

        return texts, None, ids

def create_dataloaders(data_dir,batch_size):
# 数据目录
    prepare_csv_list()

    # 数据加载
    print("📊 加载真实Kaggle数据...")

    # 为了快速训练，限制样本数（可以根据需要调整）
    train_texts, train_labels, train_ids = read_toxic_comments_real(
        data_dir, max_samples=None, is_train=True
    )

    # 创建验证集（从训练数据中分割）
    val_split = int(len(train_texts) * 0.8)
    val_texts = train_texts[val_split:]
    val_labels = train_labels[val_split:]
    train_texts = train_texts[:val_split]
    train_labels = train_labels[:val_split]

    # 读取测试数据
    test_texts, _, test_ids = read_toxic_comments_real(
        data_dir, max_samples=None, is_train=False
    )

    print(f"\n📊 数据统计:")
    print(f"训练数据: {len(train_texts)} 条")
    print(f"验证数据: {len(val_texts)} 条")
    print(f"测试数据: {len(test_texts)} 条")

    # 检查数据质量
    print(f"\n📝 数据样例:")
    print(f"文本长度: {len(train_texts[0])}")
    print(f"前100字符: {train_texts[0][:100]}")
    print(f"标签: {train_labels[0]}")

    preprocessor = TextPreprocessor()

    print(f"\n🔧 预处理器测试:")
    sample_sequence = preprocessor.text_to_sequence(train_texts[0])
    print(f"序列长度: {len(sample_sequence)}")
    print(f"非零token数: {sum(1 for x in sample_sequence if x != 0)}")



    # 创建数据加载器
    train_dataset = ToxicCommentDataset(train_texts, train_labels, preprocessor)
    val_dataset = ToxicCommentDataset(val_texts, val_labels, preprocessor)
    test_dataset = ToxicCommentDataset(test_texts, None, preprocessor)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=2, pin_memory=True, persistent_workers=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=2, pin_memory=True, persistent_workers=True)
    return train_iter, val_iter, test_iter,test_ids


    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Toxic Comment Classification - gpt2 版本 多标签文本分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from collections import Counter
import re
import os
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy: {np.__version__}")


class GPT2ClassificationModel(nn.Module):
  def __init__(self,num_labels = 6):
    super().__init__()
    self.gpt2 = GPT2Model.from_pretrained('gpt2')
    config = self.gpt2.config
    self.classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

  def forward(self,input_ids,attention_mask):
    gpt2_out = self.gpt2(input_ids,attention_mask=attention_mask)
    logits = self.classifier(gpt2_out.last_hidden_state[:, -1, :])
    return logits

# ====== 解冻控制：只解冻顶层若干层 ======
def unfreeze_gpt2_top_k_layers(model: GPT2ClassificationModel, k: int) -> None:
    """冻结 GPT-2 全部层后，仅解冻顶层 k 个 block（以及最终层归一化与分类头）。

    Args:
        model: 包含 GPT-2 的分类模型
        k: 需要解冻的顶层 block 数量（k<=总层数）。k<=0 表示只训练分类头
    """
    # 冻结全部 GPT-2 参数
    for p in model.gpt2.parameters():
        p.requires_grad = False

    # 仅解冻顶层 k 个 block
    if k > 0:
        total_blocks = len(model.gpt2.h)
        k = min(k, total_blocks)
        for block in model.gpt2.h[-k:]:
            for p in block.parameters():
                p.requires_grad = True
        # 同时解冻最终层归一化，有助于适配下游任务
        for p in model.gpt2.ln_f.parameters():
            p.requires_grad = True

    # 始终训练分类头
    for p in model.classifier.parameters():
        p.requires_grad = True

# 工具类
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        import time
        self.time = time
        self.start_time = self.time.time()
    def stop(self):
        return self.time.time() - self.start_time

def try_all_gpus():
    """检测可用GPU"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def multilabel_accuracy(y_hat, y):
    """多标签分类准确率"""
    predictions = torch.sigmoid(y_hat) > 0.5
    y = y.bool()
    label_wise_acc = (predictions == y).float().mean()
    return label_wise_acc.item()

def train_gpt2_epoch(net, train_iter, loss, updater, device, scheduler=None,progress_bar=None):
    """
    单个epoch训练 - 混合精度训练 + 学习率调度
    """
    net.train()
    metric = Accumulator(3)  # 训练损失总和, 准确数, 样本数

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    start_time = time.time()
    for _, batch in enumerate(train_iter):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 混合精度前向传播
        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_hat = net(input_ids, attention_mask)
                l = loss(y_hat, labels)
        else:
            y_hat = net(input_ids, attention_mask)
            l = loss(y_hat, labels)

        updater.zero_grad()

        # 混合精度反向传播
        if scaler is not None:
            scaler.scale(l.sum()).backward()
            scaler.unscale_(updater)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(updater)
            scaler.update()
        else:
            l.sum().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            updater.step()

        # 学习率调度（OneCycleLR需要在每个batch后调用）
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            acc = multilabel_accuracy(y_hat, labels)
            metric.add(l.sum(), acc * labels.shape[0], labels.shape[0])

        cost = time.time() - start_time
        if progress_bar is not None:
            progress_bar.set_postfix({"Cost": f"{cost:.2f}s"})

    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_gpt2_accuracy(net, data_iter, device):
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for batch in data_iter:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 使用混合精度推理
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    y_hat = net(input_ids, attention_mask)
            else:
                y_hat = net(input_ids, attention_mask)

            acc = multilabel_accuracy(y_hat, labels)
            metric.add(acc * labels.shape[0], labels.shape[0])
    return metric[0] / metric[1]

def train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, devices, scheduler=None):
    """
    完整训练流程
    """
    print('training on', devices)

    if isinstance(devices, list) and len(devices) > 1:
        # 多GPU
        net = nn.DataParallel(net, device_ids=devices)

    device = devices[0] if isinstance(devices, list) else devices
    net = net.to(device)

    for epoch in range(num_epochs):
        train_iter_tqdm = tqdm(train_iter,
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            bar_format="{desc}: {n_fmt}/{total_fmt} {postfix}")

        # 训练
        train_loss, train_acc = train_gpt2_epoch(
            net, train_iter_tqdm, loss, trainer, device, scheduler,train_iter_tqdm
        )

        # 验证
        val_acc = evaluate_gpt2_accuracy(net, val_iter, device)

        tqdm.write(f'Epoch {epoch + 1}: '
              f'loss {train_loss:.3f}, '
              f'train acc {train_acc:.3f}, '
              f'val acc {val_acc:.3f}, '
              f'lr {trainer.param_groups[0]["lr"]:.6f}')

    # print(f'Training completed in {timer.stop():.1f} sec')
    print(f'Final: train acc {train_acc:.3f}, val acc {val_acc:.3f}')


# ============ 主要执行代码 ============
print("🚀 启动双向GPT2多标签分类训练")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 使用设备: {device}')

# 模型参数 - 优化以提升训练速度
num_classes = 6  # 6个类别：toxic, severe_toxic, obscene, threat, insult, identity_hate
# 根据GPU情况自动调整批次大小
batch_size = 32 if torch.cuda.is_available() else 16
num_steps = 128   # 序列长度（从128降到64）
lr = 2e-3        # 提高学习率以加快收敛
num_epochs = 3   # 训练轮数

train_iter, val_iter, test_iter, test_ids =  create_dataloaders(data_dir,batch_size)

net = GPT2ClassificationModel()
print(f"模型参数数量: {sum(p.numel() for p in net.parameters()):,}")
net.to(device)

# --- 解冻策略：只解冻顶层若干层（其余冻结） ---
UNFREEZE_TOP_K = 2  # 修改此值控制解冻的顶层 block 数量；0 表示仅训练分类头
unfreeze_gpt2_top_k_layers(net, k=UNFREEZE_TOP_K)
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params:,} (解冻顶层 {UNFREEZE_TOP_K} 层)")
# ------------------------------------


# 优化器和学习率调度器（只优化可训练参数）
trainer = optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)

# 添加学习率调度器以提升训练效果
scheduler = optim.lr_scheduler.OneCycleLR(
    trainer,
    max_lr=lr * 5,  # 最大学习率
    steps_per_epoch=len(train_iter),
    epochs=num_epochs,
    pct_start=0.3  # 前30%时间用于升温
)

# 损失函数 - 多标签分类使用BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss(reduction="none")  # 每个样本每个标签独立计算


# 开始训练 - 使用学习率调度器
# train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, device, scheduler)

print("\n" + "="*60)
print("🎉 Huggingface GPT2 多标签分类 训练完成!")


import time
from tqdm.std import tqdm
def generate_submission(model, test_loader, device, test_ids, output_path):
    """
    生成Kaggle提交文件
    """
    model.eval()
    predictions = []

    print("🔮 生成预测结果...")
    with torch.no_grad():
        start_time = time.time()
        test_loader_tqdm = tqdm(test_loader,bar_format=" {n_fmt}/{total_fmt} {postfix}")

        for batch in test_loader_tqdm:
            try:
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                # 使用混合精度推理
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids, attention_mask)

                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs)
                cost = time.time() - start_time
                test_loader_tqdm.set_postfix({"Cost": f"{cost:.2f}s"})
            except Exception as e:
                print(f"❌ 在 batch {i} 报错：{repr(e)}")
                print(f"batch 内容信息:")
                for j, item in enumerate(batch):
                    if torch.is_tensor(item):
                        print(f"  tensor[{j}] -> shape: {item.shape}, dtype: {item.dtype}")
                    else:
                        print(f"  非tensor[{j}]: {type(item)}")
                raise e


    # 创建提交DataFrame
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submission_df = pd.DataFrame({
        'id': test_ids,
        **{col: [pred[i] for pred in predictions] for i, col in enumerate(label_columns)}
    })

    # 保存提交文件
    submission_df.to_csv(output_path, index=False)
    print(f"💾 提交文件已保存: {output_path}")
    print(f"📊 预测统计:")
    for i, col in enumerate(label_columns):
        avg_prob = sum(pred[i] for pred in predictions) / len(predictions)
        print(f"  {col}: 平均概率 {avg_prob:.4f}")

    return submission_df

    # 生成提交文件
submission_path = os.path.join(data_dir, 'submission.csv')
submission_df = generate_submission(net, test_iter, device, test_ids, submission_path)

print("\n🎉 训练和预测完成!")
print(f"✅ 提交文件: {submission_path}")