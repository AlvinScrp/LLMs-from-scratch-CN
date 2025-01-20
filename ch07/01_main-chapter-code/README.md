# 第7章：微调以跟随指令

### 主要章节代码

- [ch07.ipynb](ch07.ipynb) 包含本章中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个Python模块，包含我们在前几章中编码和训练的GPT模型以及许多实用功能，我们在本章中重用这些功能
- [gpt_download.py](gpt_download.py) 包含下载预训练GPT模型权重的实用函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的习题解答

### 可选代码

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) 是一个独立的Jupyter笔记本，用于加载我们在本章创建的指令微调模型
- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) 是一个独立的Python脚本，用于按本章所述进行指令微调（可以视为本章微调部分的总结）

用法：
```bash
python gpt_instruction_finetuning.py
```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) 是一个独立的Python脚本，用于评估微调模型的响应，如本章所述（可以视为本章评估部分的总结）

用法：
```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) 是一个可选的脚本，实现了习题解答；更多细节请见 [exercise-solutions.ipynb](exercise-solutions.ipynb)
