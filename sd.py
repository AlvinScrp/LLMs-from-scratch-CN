import os
import urllib.request
from pathlib import Path
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import time
import warnings
import datasets

warnings.filterwarnings('ignore')

# 全局配置
URLPrefix = "https://pro-5gu0t2os8cdd45f2-1251420592.tcloudbaseapp.com/toxic-comment-classification"
data_dir = 'toxic-comment'
DATA_DIR = Path(data_dir)
FILENAMES = ["train.csv","test.csv","test_labels.csv","sample_submission.csv"]


def prepare_csv_list():
    """下载数据文件"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fileName in FILENAMES:
        URL = f"{URLPrefix}/{fileName}"
        DATA_FILE = DATA_DIR / fileName
        if not DATA_FILE.exists():
            print(f"⬇️ Downloading {fileName}...")
            with urllib.request.urlopen(URL) as r, open(DATA_FILE, "wb") as f:
                f.write(r.read())
        else:
            print(f"✅ already exists: {fileName}")


def create_dataloaders(data_dir, batch_size, max_length=128):
    """创建数据加载器"""
    prepare_csv_list()
    
    # 读取数据
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # 分割验证集
    val_split = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_split:].copy()
    train_df = train_df.iloc[:val_split].copy()
    
    print(f"📊 数据统计: 训练{len(train_df)} 验证{len(val_df)} 测试{len(test_df)}")
    
    # 初始化分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 预处理函数
    def preprocess_batch(examples):
        texts = [str(text).strip() if not pd.isna(text) else "" for text in examples['comment_text']]
        tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors=None)
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}
    
    # 创建数据集
    train_dataset = datasets.Dataset.from_pandas(train_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=4)
    val_dataset = datasets.Dataset.from_pandas(val_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=4)
    test_dataset = datasets.Dataset.from_pandas(test_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=4)
    
    # 添加标签
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    def add_labels(examples):
        return {'labels': [[examples[col][i] for col in label_cols] for i in range(len(examples['id']))]}
    
    train_dataset = train_dataset.map(add_labels, batched=True)
    val_dataset = val_dataset.map(add_labels, batched=True)
    
    # 设置格式
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # 创建数据加载器 (优化: 增加工作进程和预取因子)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    return train_loader, val_loader, test_loader, test_df['id'].tolist()


class GPT2ClassificationModel(nn.Module):
  def __init__(self,num_labels = 6):
    super().__init__()
    self.gpt2 = GPT2Model.from_pretrained('gpt2')
    config = self.gpt2.config
    self.dropout = nn.Dropout(p=0.1)
    self.classifier = nn.Linear(config.hidden_size, num_labels, bias=True)

  def forward(self,input_ids,attention_mask):
    gpt2_out = self.gpt2(input_ids,attention_mask=attention_mask)
    hidden = gpt2_out.last_hidden_state  # [B, T, H]
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    masked_sum = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    pooled = masked_sum / denom
    logits = self.classifier(self.dropout(pooled))
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

def move_batch_to_device(batch, device, has_labels=True):
    """将批次数据移动到指定设备"""
    input_ids, attention_mask = batch[:2]
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    
    if has_labels and len(batch) > 2:
        labels = batch[2].to(device, non_blocking=True)
        return input_ids, attention_mask, labels
    else:
        return input_ids, attention_mask



def multilabel_accuracy(y_hat, y):
    """多标签分类准确率"""
    predictions = torch.sigmoid(y_hat) > 0.5
    y = y.bool()
    label_wise_acc = (predictions == y).float().mean()
    return label_wise_acc.item()

def evaluate_model_metrics(net, data_iter, device):
    """评估模型指标：收集预测结果并计算AUC和F1分数"""
    net.eval()
    all_probs, all_labels = [], []
    
    # 收集预测结果
    with torch.no_grad():
        for batch in data_iter:
            input_ids, attention_mask, labels = move_batch_to_device(batch, device)
            
            with torch.cuda.amp.autocast():
                logits = net(input_ids, attention_mask)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # 计算AUC和F1分数
    num_labels = probs.shape[1]
    best_thresholds, per_label_f1, aucs = [], [], []
    
    for j in range(num_labels):
        # AUC计算
        try:
            auc = roc_auc_score(labels[:, j], probs[:, j])
        except Exception:
            auc = float('nan')
        aucs.append(auc)
        
        # 最佳阈值和F1分数
        thrs = np.linspace(0.05, 0.95, 37)
        best_t, best_f1 = 0.5, -1.0
        y_true, p = labels[:, j], probs[:, j]
        
        for t in thrs:
            y_pred = (p >= t).astype(int)
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        
        best_thresholds.append(best_t)
        per_label_f1.append(best_f1)
    
    # 宏平均
    aucs_np = np.array(aucs, dtype=float)
    macro_auc = float(np.nanmean(aucs_np)) if np.isnan(aucs_np).any() else float(aucs_np.mean())
    macro_f1 = float(np.mean(per_label_f1))
    
    return probs, labels, macro_auc, macro_f1, best_thresholds

def train_gpt2_epoch(net, train_iter, loss, updater, device, scheduler=None, progress_bar=None, accumulation_steps=1):
    """
    单个epoch训练 - 混合精度训练 + 学习率调度 + 梯度累积
    """
    net.train()
    metric = Accumulator(3)  # 训练损失总和, 准确数, 样本数

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    for batch_idx, batch in enumerate(train_iter):
        input_ids, attention_mask, labels = move_batch_to_device(batch, device)

        # 混合精度前向传播
        with torch.cuda.amp.autocast():
            y_hat = net(input_ids, attention_mask)
            l = loss(y_hat, labels)
            # 梯度累积缩放
            l = l / accumulation_steps

        # 反向传播
        scaler.scale(l.sum()).backward()

        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(updater)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(updater)
            scaler.update()
            updater.zero_grad()

            # 学习率调度（OneCycleLR需要在每个batch后调用）
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            acc = multilabel_accuracy(y_hat, labels)
            metric.add(l.sum() * accumulation_steps, acc * labels.shape[0], labels.shape[0])

        cost = time.time() - start_time
        if progress_bar is not None:
            progress_bar.set_postfix({"Cost": f"{cost:.2f}s"})

    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_gpt2_accuracy(net, data_iter, device):
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for batch in data_iter:
            input_ids, attention_mask, labels = move_batch_to_device(batch, device)

            # 使用混合精度推理
            with torch.cuda.amp.autocast():
                y_hat = net(input_ids, attention_mask)

            acc = multilabel_accuracy(y_hat, labels)
            metric.add(acc * labels.shape[0], labels.shape[0])
    return metric[0] / metric[1]

def train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, devices, scheduler=None, patience=2):
    """
    完整训练流程
    """
    print('training on', devices)

    if isinstance(devices, list) and len(devices) > 1:
        # 多GPU
        net = nn.DataParallel(net, device_ids=devices)

    device = devices[0] if isinstance(devices, list) else devices
    net = net.to(device)

    best_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        train_iter_tqdm = tqdm(train_iter,
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            bar_format="{desc}: {n_fmt}/{total_fmt} {postfix}")

        # 训练 (添加梯度累积)
        train_loss, train_acc = train_gpt2_epoch(
            net, train_iter_tqdm, loss, trainer, device, scheduler, train_iter_tqdm, GRADIENT_ACCUMULATION_STEPS
        )

        # 验证
        val_probs, val_labels, macro_auc, macro_f1, best_thrs = evaluate_model_metrics(net, val_iter, device)

        tqdm.write(
            f'Epoch {epoch + 1}: '
            f'loss {train_loss:.3f}, '
            f'train acc {train_acc:.3f}, '
            f'val macro AUC {macro_auc:.4f}, '
            f'val macro F1 {macro_f1:.4f}, '
            f'lr {trainer.param_groups[0]["lr"]:.6f}'
        )

        # Early stopping on macro AUC
        if macro_auc > best_auc:
            best_auc = macro_auc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1} (best macro AUC {best_auc:.4f})")
                break

    # print(f'Training completed in {timer.stop():.1f} sec')
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f'Final: best val macro AUC {best_auc:.4f}')


# 主执行代码
print("🚀 启动GPT2多标签分类训练")

# 配置参数 - 可根据需要调整
MAX_LENGTH = 128  # 最大序列长度
BATCH_SIZE = 64   # 批次大小 (优化: 32 → 64)
NUM_EPOCHS = 3    # 训练轮数
UNFREEZE_LAYERS = 2  # 解冻的顶层层数
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = BATCH_SIZE if torch.cuda.is_available() else 16
num_epochs = NUM_EPOCHS

print(f"📊 配置信息:")
print(f"  最大序列长度: {MAX_LENGTH}")
print(f"  批次大小: {batch_size}")
print(f"  训练轮数: {num_epochs}")
print(f"  解冻层数: {UNFREEZE_LAYERS}")
print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")

# 数据加载
train_iter, val_iter, test_iter, test_ids = create_dataloaders(data_dir, batch_size, MAX_LENGTH)

# 模型
net = GPT2ClassificationModel()
net.to(device)
unfreeze_gpt2_top_k_layers(net, k=UNFREEZE_LAYERS)

# 模型编译优化 (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    try:
        net = torch.compile(net, mode="reduce-overhead")
        print("✅ 启用模型编译优化")
    except Exception as e:
        print(f"⚠️  模型编译失败: {e}")

# 梯度检查点优化
try:
    net.gpt2.gradient_checkpointing_enable()
    print("✅ 启用梯度检查点")
except Exception:
    pass

print(f"可训练参数: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

# 优化器
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 优化器配置
base_params = [p for n, p in net.named_parameters() if p.requires_grad and "classifier" not in n]
head_params = [p for n, p in net.named_parameters() if p.requires_grad and "classifier" in n]

trainer = AdamW([
    {"params": base_params, "lr": 5e-5},
    {"params": head_params, "lr": 1e-3},
], weight_decay=0.01)

# 学习率调度
num_training_steps = len(train_iter) * num_epochs
num_warmup_steps = max(1, int(0.1 * num_training_steps))
scheduler = get_linear_schedule_with_warmup(trainer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# 损失函数
train_labels_array = np.array(train_iter.dataset.labels)
pos = train_labels_array.sum(axis=0)
neg = len(train_labels_array) - pos
pos_weight = torch.tensor((neg / (pos + 1e-6)).tolist(), dtype=torch.float).to(device)
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

# 训练
net.gpt2.config.use_cache = False
try:
    net.gpt2.gradient_checkpointing_enable()
except Exception:
    pass

train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, device, scheduler)
print("🎉 训练完成!")


import time
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

        for i, batch in enumerate(test_loader_tqdm):
            try:
                input_ids, attention_mask = move_batch_to_device(batch, device, has_labels=False)

                # 使用混合精度推理
                with torch.cuda.amp.autocast():
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
print(f"✅ 提交文件: {submission_path}")