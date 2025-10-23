# GPT-2 有毒评论分类（sd.py）

本文档总结了对 `sd.py` 的改进点、运行方式，以及如何在 Kaggle Jigsaw Toxic Comment Classification Challenge 上进一步提升效果。

## 1) 本次改进

- 修正 attention mask：明确将 padding 位置排除在注意力与池化之外。
- 使用掩码均值池化：替代“最后一个 token”池化，句子表征更稳定。
- AdamW + 参数分组学习率：骨干较小 LR、分类头较大 LR。
- 线性 warmup + 线性衰减调度器。
- 类别不平衡处理：为每个标签计算 `pos_weight` 用于 `BCEWithLogitsLoss`。
- 验证指标与阈值调优：输出 macro AUC、macro‑F1，并在验证集上为每个标签搜索最优阈值。
- 早停：以 macro AUC 为准并回滚到最佳权重。
- 稳定性与显存：开启 gradient checkpointing，禁用 `use_cache`。
- 实际训练已开启；推理后生成提交 CSV。

## 2) 运行环境

安装依赖：

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm
```

建议使用 Python 3.9 及以上版本。

## 3) 数据

脚本会在缺失时自动下载 Kaggle 数据到 `toxic-comment/`：
- `train.csv`、`test.csv`、`test_labels.csv`、`sample_submission.csv`

无需手动预处理。

## 4) 运行方式

```bash
python sd.py
```

流程概览：
- 构建 DataLoader，按 80/20 划分训练/验证
- 初始化 GPT‑2 分类器，解冻顶层若干 block
- 使用 AdamW + 线性 warmup/decay 与 `pos_weight` 训练
- 每轮验证并输出 macro AUC / macro‑F1；基于 AUC 早停
- 在测试集上推理并生成 `toxic-comment/submission.csv`

## 5) 关键可调参数（在 `sd.py` 中修改）

- 解冻深度：`UNFREEZE_TOP_K`
  - 建议 6–12；更大通常更准，但计算与显存开销更高
- 序列长度：`TextPreprocessor(..., max_seq_length=...)`
  - 若显存允许，256 或 384 往往更好
- 训练轮数：`num_epochs`
  - 一般 5–8，已启用早停
- 学习率（参数分组）：
  - 骨干：`3e-5` ~ `1e-4`
  - 分类头：`5e-4` ~ `1e-3`
- 批大小：视显存调大；不足时保持现状

## 6) 指标与阈值

- 主要验证指标为 macro AUC（与 Kaggle 评分一致）与 macro‑F1
- 在验证集上进行每标签阈值搜索；这些阈值能提升 Accuracy/F1，但 Kaggle 的 AUC 与阈值无关

## 7) 提分建议

- 增大 `max_seq_length` 以保留更多上下文
- 解冻更多 GPT‑2 层，或尝试全量微调（骨干 LR 更小）
- 保持开启 gradient checkpointing 节省显存
- 若 AUC 停滞，降低骨干 LR 一档并适度延长训练

## 8) 输出

- 日志：训练损失、训练准确率、macro AUC、macro‑F1、学习率
- 早停提示：在验证指标无提升时提前停止
- 提交文件：`toxic-comment/submission.csv`

## 9) 排错

- 编辑器中的导入告警通常源于缺依赖；先执行上面的 `pip install`
- 若出现 CUDA OOM，调小 `batch_size` 或 `max_seq_length`
- 若验证 AUC 不升，尝试增大 `UNFREEZE_TOP_K` 并降低骨干 LR

## 10) 说明

- 该竞赛的主指标是 AUC（与阈值无关）。请优先关注日志中的 macro AUC 以评估真实进步。
- 在该不平衡数据集上，Accuracy 往往虚高；比较模型时更建议看 AUC/macro‑F1。
