[📖 English ReadMe](./README.md)

## Introduction
在本 BERT 实现中，我们展示如何在 [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) 和 [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 数据集上进行预训练，随后加载 Hugging Face 提供的官方预训练权重，并在 [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) 数据集上进行微调，以复现论文报告的实验结果。

## Model Details

### Key Differences with Vanilla Transformer and GPT

BERT 的许多设计选择旨在使其尽可能接近原始 GPT，从而支持对两种方法进行最小差异比较。BERT_base 的模型规模、注意力头数和层数均与 GPT 保持一致；同时，BERT 将激活函数由 ReLU 替换为 GeLU，并采用可学习的位置嵌入而非正余弦位置编码。两者的主要差异如下：

1. **BERT 的自注意力机制**：BERT 通过双向自注意力来捕捉上下文信息，而 GPT 仅使用因果自注意力。换而言之，BERT 的 mask 只用于 padding。
2. **BERT 的词嵌入**：BERT 使用的是 WordPiece 的分词方式，这与 GPT 的 Byte-Pair Encoding（BPE）不同。
3. **训练目标**：BERT 使用了 Masked Language Modeling (MLM) 和 Next Sentence Prediction (NSP) 作为预训练目标，而 GPT 和原始 Transformer 采用的是传统的语言建模目标。
4. **token 处理**：BERT 在预训练阶段即学习 [SEP]、[CLS] 以及句子 A/B 的嵌入；GPT 虽然也使用 [SEP] 和 [CLS]，但这些特殊 token 仅在微调阶段引入。

### Pre-training Tasks

#### Masked Language Modeling (MLM)

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

##### Rules:

对于每个输入序列，随机选取 15% 的 token，并按照以下规则进行替换：

1. 80% 的被选中 token 替换为 `[MASK]`。
2. 10% 的被选中 token 替换为随机 token。
3. 10% 的被选中 token 保持不变，但仍纳入预测目标。

#### Next Sentence Prediction (NSP)

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

该任务用于建模两个文本片段之间的关系，而这种关系并不能由标准语言建模目标直接捕获。

##### Rules:

1. 50% 的样本中，第二个句子是第一个句子的真实后续句。
2. 其余 50% 的样本中，第二个句子来自不相关的上下文。

## [Pre-training](./pretrain.ipynb)
根据 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 论文中的设置，BERT 在 BooksCorpus 和Wikipedia 数据集上进行预训练，使用 AdamW 优化器（$w = 0.01, \text{max-lr} = 1 \times 10^{-4}$）。训练时使用了线性增长的学习率策略，学习率在前 10000 步线性增加，之后使用线性衰减策略调整学习率。

## [Fine-tuning](./finetune.ipynb)
预训练完成后，BERT 已经获得了较强的语言理解能力，可以通过微调来适应新的任务。在微调时，只需要对模型结构做轻微调整，并在下游任务中添加适当的分类头。

由于最佳超参数值取决于具体任务，原论文针对不同任务的微调给出了超参数范围：

- **Batch size**：16，32
- **Learning rate **：5e-5, 3e-5, 2e-5
- **Number of epochs**: 2, 3, 4

在本实现中，我们使用了 AdamW 优化器（$w = 0.01, \text{max-lr} = 4 \times 10^{-5}$），并选用 batch size 为 32 ，epoch 数为 3 。微调同样使用了线性增长的学习率策略，学习率在前 10000 步线性增加，之后使用线性衰减策略调整学习率。

## Appendix
### How to Download Pretrained BERT?
在命令行运行以下指令：
```bash
pip install -U huggingface-cli
huggingface-cli download bert-base-uncased --local-dir path/to/pretrained_dir
```
