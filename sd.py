import os
import urllib.request
from pathlib import Path
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, DataCollatorWithPadding
from tqdm import tqdm
import time
import warnings
import datasets

warnings.filterwarnings('ignore')

# ========= 文本清洗规则（来自：优化数据comment_text.md） =========
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
# We will filter all characters except alphabet characters and some punctuation
valid_characters = " " + "@$" + "'!?-" + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
valid_characters_ext = valid_characters + "abcdefghijklmnopqrstuvwxyz".upper()

# 直接判定词（当前不直接用于打标签，仅用于清洗参考）
TOXIC_WORDS = [
    "poop", "crap", "prick", "twat", "wikipedia", "wiki", "hahahahaha", "lol",
    "bastard", "sluts", "slut", "douchebag", "douche", "blowjob", "nigga", "dumb",
    "jerk", "wanker", "wank", "penis", "motherfucker", "fucker", "fuk", "fucking",
    "fucked", "fuck", "bullshit", "shit", "stupid", "bitches", "bitch", "suck",
    "cunt", "dick", "cocks", "cock", "die", "kill", "gay", "jewish", "jews", "jew",
    "niggers", "nigger", "faggot", "fag", "asshole"
]

# 星号替换词对（前者→后者）
ASTERICKS_WORDS = [
    ('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'),
    ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'),
    ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'),
    ('fu*cking', 'fucking'), ('****ing', 'fucking'), ('a**hole', 'asshole'),
    ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'),
    ('f***ing', 'fucking'), ('f**king', 'fucking'), ('f*cking', 'fucking'),
    ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'),
    ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'),
    ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'),
    ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'),
    ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'), ('f**ks', 'fucks'),
    ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'),
    ('cr*p', 'crap'), ('d*ck', 'dick'), ('f***', 'fuck'), ('f**k', 'fuck'),
    ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'),
    ('s***', 'shit'), ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')
]

# fastText 拼写归一
FASTTEXT_MISSPELLINGS = {"'n'balls": 'balls', "-nazi's": 'nazis', 'adminabuse': 'admin abuse', "admins's": 'admins', 'arsewipe': 'arse wipe', 'assfack': 'asshole', 'assholifity': 'asshole', 'assholivity': 'asshole', 'asshoul': 'asshole', 'asssholeee': 'asshole', 'belizeans': 'mexicans', "blowing's": 'blowing', 'bolivians': 'mexicans', 'celtofascists': 'fascists', 'censorshipmeisters': 'censor', 'chileans': 'mexicans', 'clerofascist': 'fascist', 'cowcrap': 'crap', 'crapity': 'crap', "d'idiots": 'idiots', 'deminazi': 'nazi', 'dftt': "don't feed the troll", 'dildohs': 'dildo', 'dramawhores': 'drama whores', 'edophiles': 'pedophiles', 'eurocommunist': 'communist', 'faggotkike': 'faggot', 'fantard': 'retard', 'fascismnazism': 'fascism', 'fascistisized': 'fascist', 'favremother': 'mother', 'fuxxxin': 'fucking', "g'damn": 'goddamn', 'harassmentat': 'harassment', 'harrasingme': 'harassing me', 'herfuc': 'motherfucker', 'hilterism': 'fascism', 'hitlerians': 'nazis', 'hitlerites': 'nazis', 'hubrises': 'pricks', 'idiotizing': 'idiotic', 'inadvandals': 'vandals', "jackass's": 'jackass', 'jiggabo': 'nigga', 'jizzballs': 'jizz balls', 'jmbass': 'dumbass', 'lejittament': 'legitimate', "m'igger": 'nigger', "m'iggers": 'niggers', 'motherfacking': 'motherfucker', 'motherfuckenkiwi': 'motherfucker', 'muthafuggas': 'niggas', 'nazisms': 'nazis', 'netsnipenigger': 'nigger', 'niggercock': 'nigger', 'niggerspic': 'nigger', 'nignog': 'nigga', 'niqqass': 'niggas', "non-nazi's": 'not a nazi', 'panamanians': 'mexicans', 'pedidiots': 'idiots', 'picohitlers': 'hitler', 'pidiots': 'idiots', 'poopia': 'poop', 'poopsies': 'poop', 'presumingly': 'obviously', 'propagandaanddisinformation': 'propaganda and disinformation', 'propagandaministerium': 'propaganda', 'puertoricans': 'mexicans', 'puertorricans': 'mexicans', 'pussiest': 'pussies', 'pussyitis': 'pussy', 'rayaridiculous': 'ridiculous', 'redfascists': 'fascists', 'retardzzzuuufff': 'retard', "revertin'im": 'reverting', 'scumstreona': 'scums', 'southamericans': 'mexicans', 'strasserism': 'fascism', 'stuptarded': 'retarded', "t'nonsense": 'nonsense', "threatt's": 'threat', 'titoists': 'communists', 'twatbags': 'douchebags', 'youbollocks': 'you bollocks'}

# 收缩词展开
CONT_PATTERNS = [
    (r'(W|w)on\'t', r'will not'),
    (r'(C|c)an\'t', r'can not'),
    (r'(I|i)\'m', r'i am'),
    (r'(A|a)in\'t', r'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),
]

def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = s.strip().lower()
    # 处理不间断空格 \xa0
    s = s.replace("\xa0", " ")

    # 展开收缩词
    for pattern, repl in CONT_PATTERNS:
        s = re.sub(pattern, repl, s)

    # 星号变体归一
    for bad, good in ASTERICKS_WORDS:
        s = s.replace(bad, good)

    # fastText 拼写归一（按词边界）
    for bad, good in FASTTEXT_MISSPELLINGS.items():
        s = re.sub(rf'\b{re.escape(bad)}\b', good, s)

    # 去除特殊字符（替换为空格，保留分词边界）
    trans = {ord(c): " " for c in CHARS_TO_REMOVE}
    s = s.translate(trans)

    # 允许字符白名单过滤（不在白名单的字符替换为空格）
    allowed = set(valid_characters_ext)
    s = ''.join(ch if ch in allowed else ' ' for ch in s)

    # 合并多空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    usecols_train = ['id', 'comment_text'] + label_cols
    usecols_test = ['id', 'comment_text']
    train_df = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols_train)
    test_df = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols_test)
    
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
        texts = [normalize_text(text) for text in examples['comment_text']]
        # 动态 padding：此处不填充，只截断；由 collator 在 DataLoader 内按 batch 最长填充
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}
    
    # 创建数据集（可并行映射）
    cpu_cnt = os.cpu_count() or 2
    MAP_NUM_PROC = min(4, max(1, cpu_cnt // 2))  # 适度并行，避免过载
    train_dataset = datasets.Dataset.from_pandas(train_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=MAP_NUM_PROC)
    val_dataset = datasets.Dataset.from_pandas(val_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=MAP_NUM_PROC)
    test_dataset = datasets.Dataset.from_pandas(test_df).map(preprocess_batch, batched=True, batch_size=1000, num_proc=MAP_NUM_PROC)
    
    # 添加标签
    # 添加标签
    def add_labels(examples):
        labels = [[float(examples[col][i]) for col in label_cols] for i in range(len(examples['id']))]
        return {'labels': labels}
    
    train_dataset = train_dataset.map(add_labels, batched=True)
    val_dataset = val_dataset.map(add_labels, batched=True)
    
    # 仅保留模型需要的列，避免 collator 看到字符串/列表等无法转张量的字段
    keep_train_cols = ['input_ids', 'attention_mask', 'labels']
    keep_val_cols = ['input_ids', 'attention_mask', 'labels']
    keep_test_cols = ['input_ids', 'attention_mask']
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_train_cols])
    val_dataset = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in keep_val_cols])
    test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_test_cols])

    # 不将数据集预先转换为 torch；交由 collator 动态 padding 并转换
    
    # DataCollator：按 batch 最长动态 padding，8 对齐以加速 Tensor Core
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', pad_to_multiple_of=8, return_tensors='pt')

    def collate_fn(features):
        labels = None
        if 'labels' in features[0]:
            labels = [f['labels'] for f in features]
            for f in features:
                f.pop('labels', None)
        batch = collator(features)
        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.float)
        return batch

    # 创建数据加载器（单进程，更稳）
    NUM_WORKERS_TRAIN = 0
    NUM_WORKERS_EVAL = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"🧵 DataLoader workers -> train/val/test: 0")
    
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
    pass

def try_all_gpus():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def move_batch_to_device(batch, device, has_labels=True):
    """将批次数据移动到指定设备"""
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    
    if has_labels and 'labels' in batch:
        labels = batch['labels'].to(device, non_blocking=True)
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
    """评估（基于准确率）并返回概率与标签，方便后续需要时扩展。"""
    net.eval()
    all_probs, all_labels = [], []
    metric = Accumulator(2)
    with torch.no_grad():
        for batch in data_iter:
            input_ids, attention_mask, labels = move_batch_to_device(batch, device)
            logits = net(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            # 准确率（0.5 阈值）
            acc = multilabel_accuracy(logits, labels.float())
            metric.add(acc * labels.shape[0], labels.shape[0])
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    val_acc = metric[0] / metric[1]
    return probs, labels, val_acc

def train_gpt2_epoch(net, train_iter, loss, updater, device,  progress_bar=None,  log_interval=50):
    """
    单个epoch训练 - 混合精度训练 + 学习率调度 + 梯度累积
    """
    net.train()
    metric = Accumulator(3)  # 训练损失总和, 准确数, 样本数

    start_time = time.time()
    for batch_idx, batch in enumerate(train_iter):
        input_ids, attention_mask, labels = move_batch_to_device(batch, device)

        # 前向传播（不使用 autocast）
        y_hat = net(input_ids, attention_mask)
        labels_hard = labels.float()
        l = loss(y_hat, labels_hard)

        # 反向传播
        l.sum().backward()

        # 每个 batch 都进行一次优化步
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        updater.step()
        updater.zero_grad()

        with torch.no_grad():
            acc = multilabel_accuracy(y_hat, labels_hard)
            metric.add(l.sum(), acc * labels.shape[0], labels.shape[0])

        cost = time.time() - start_time
        if progress_bar is not None and ((batch_idx + 1) % log_interval == 0):
            progress_bar.set_postfix({"Cost": f"{cost:.2f}s"})

    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_gpt2_accuracy(net, data_iter, device):
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for batch in data_iter:
            input_ids, attention_mask, labels = move_batch_to_device(batch, device)
            y_hat = net(input_ids, attention_mask)

            # 确保标签是浮点类型
            labels = labels.float()
            acc = multilabel_accuracy(y_hat, labels)
            metric.add(acc * labels.shape[0], labels.shape[0])
    return metric[0] / metric[1]

def train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, device, patience=3):
    """
    完整训练流程
    """
    print('training on', device)
    net = net.to(device)

    best_acc = -1.0
    best_state = None
    best_thresholds = None
    for epoch in range(num_epochs):
        train_iter_tqdm = tqdm(train_iter,
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            bar_format="{desc}: {n_fmt}/{total_fmt} {postfix}",
                            mininterval=1.0)

        # 训练 (添加梯度累积)
        train_loss, train_acc = train_gpt2_epoch(
            net, train_iter_tqdm, loss, trainer, device, train_iter_tqdm, log_interval=50
        )

        # 验证（基于准确率）
        val_probs, val_labels, val_acc = evaluate_model_metrics(net, val_iter, device)

        tqdm.write(
            f'Epoch {epoch + 1}: '
            f'loss {train_loss:.3f}, '
            f'train acc {train_acc:.3f}, '
            f'val acc {val_acc:.4f}, '
            f'lr {trainer.param_groups[0]["lr"]:.6f}'
        )

        # Early stopping & best checkpoint based on val accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1} (best val acc {best_acc:.4f})")
                break

    # print(f'Training completed in {timer.stop():.1f} sec')
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f'Final: best val acc {best_acc:.4f}')
    return None


# 主执行代码
print("🚀 启动GPT2多标签分类训练")

# 配置参数 - 可根据需要调整
max_length = 128  # 最大序列长度
batch_size = 32   # 批次大小
num_epochs = 8  # 训练轮数

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = batch_size 

print(f"📊 配置信息:")
print(f"  最大序列长度: {max_length}")
print(f"  批次大小: {batch_size}")
print(f"  训练轮数: {num_epochs}")

# 数据加载
train_iter, val_iter, test_iter, test_ids = create_dataloaders(data_dir, batch_size, max_length)

net = GPT2ClassificationModel()
net.to(device)



print(f"可训练参数: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

# 优化器
from torch.optim import AdamW

# 优化器配置
base_params = [p for n, p in net.named_parameters() if p.requires_grad and "classifier" not in n]
head_params = [p for n, p in net.named_parameters() if p.requires_grad and "classifier" in n]

trainer = AdamW([
    {"params": base_params, "lr": 5e-5},   # 更稳的骨干学习率
    {"params": head_params, "lr": 1e-3},   # 更稳的分类头学习率
    ], weight_decay=0.01, fused=True)

# 损失函数 - 从数据集中提取标签
print("📊 计算类别权重...")
# 使用批量访问提高效率
train_labels_array = np.array(train_iter.dataset['labels'])
pos = train_labels_array.sum(axis=0)
neg = len(train_labels_array) - pos
pos_weight = torch.tensor((neg / (pos + 1e-6)).tolist(), dtype=torch.float).to(device)
# 限制 pos_weight 上限，防止过大导致训练不稳
pos_weight = torch.clamp(pos_weight, max=10.0)
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
print(f"✅ 类别权重计算完成: {pos_weight.cpu().numpy()}")


best_thresholds = train_gpt2_model(net, train_iter, val_iter, loss, trainer, num_epochs, device)
print("🎉 训练完成!")


import time
def generate_submission(model, test_loader, device, test_ids, output_path, thresholds=None):
    model.eval()
    preds = []
    with torch.no_grad():
        start_time = time.time()
        prog = tqdm(test_loader, bar_format=" {n_fmt}/{total_fmt} {postfix}", mininterval=1.0)
        for batch in prog:
            input_ids, attention_mask = move_batch_to_device(batch, device, has_labels=False)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            elapsed = time.time() - start_time
            prog.set_postfix({"Cost": f"{elapsed:.2f}s"})

    preds_array = np.vstack(preds).astype(float)
    n = min(len(test_ids), preds_array.shape[0])
    test_ids = list(test_ids)[:n]
    preds_array = preds_array[:n]

    if thresholds is not None:
        thr = np.array(thresholds, dtype=float).reshape(1, -1)
        preds_array = (preds_array >= thr).astype(float)

    preds_array = np.nan_to_num(preds_array, nan=0.0, posinf=1.0, neginf=0.0)

    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    data = {'id': test_ids}
    for i, col in enumerate(label_columns):
        data[col] = preds_array[:, i].astype(float)
    submission_df = pd.DataFrame(data)
    submission_df.to_csv(output_path, index=False)
    return submission_df

# 生成提交文件
submission_path = os.path.join(data_dir, 'submission.csv')
submission_df = generate_submission(net, test_iter, device, test_ids, submission_path, thresholds=None)
print(f"✅ 提交文件: {submission_path}")