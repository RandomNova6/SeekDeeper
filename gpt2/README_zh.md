[📖English ReadMe](./README.md)
## Introduction
在本 GPT-2 实现中，我们展示如何在 [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) 数据集上进行预训练，随后从 [Hugging Face](https://huggingface.co/openai-community/gpt2/tree/main) 加载官方预训练权重到我们的模型中，并在 [Children's Book Test (CBT)](https://arxiv.org/pdf/1511.02301) 数据集上不经微调直接评估，以复现论文报告的实验结果。

## Model details
### Key differences with GPT-1  
1. GPT-2 将 `LayerNorm` 移至每个子块的输入，并在最后的添加了额外的 `LayerNorm`。有关 `Pre-LN` 的分析，请见论文 [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)。
2. GPT-2 将词表大小从 40478 扩大到了 50257，并且把上下文长度从 512 扩大到了 1024。  
3. GPT-2 使用了经过修改的初始化方法。这种初始化考虑了残差路径随着模型深度的累积。在初始化时，将残差映射层的权重缩放为 $1/\sqrt{N}$，其中 $N$ 是残差层的数量。

### [Byte-pair encoding (BPE)](./modules/bpe.py)  
BPE 是一种 tokenize 的方法，其核心思想是通过合并最频繁出现的字符对来构建更大的子词单元，从而减少词汇表的大小，处理稀有词问题。它需要先在一个语料库上进行训练，得到词表后才能进行编码和解码。

我们的实现主要参考 Karpathy 的 [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py)。关于 BPE 训练流程的更多细节，也可参考其 [minbpe](https://github.com/karpathy/minbpe)。

GPT-2 所使用的 BPE 相较 GPT-1 所使用的版本做了一些改进。GPT-2 的 BPE 不是像 GPT-1 那样基于 Unicode 字符进行分割，而是基于字节级别。这意味着 GPT-2 可以更加灵活地处理各种字符集和特殊符号，特别是非 ASCII 字符和表情符号等，这对于多语言支持和处理非英语文本非常有帮助。
  
## [Pre-training](./pretrain.ipynb)  
大致步骤与 GPT-1 的预训练是相同的。

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Sec. 2.1，详细介绍了 OpenWebText 是如何生成的，然而在后续段落中也没有提及太多关于训练超参数的细节。

## [Inference](./inference.ipynb)  
相较于 GPT-1，GPT-2 的文本生成能力显著提升。我们复现论文中在 Children's Book Test (CBT) 的 CN 子集上的评估。该子集要求模型在上下文段落中从十个候选名词中选择合适词项填入空缺位置，并以预测准确率作为评估指标。为完成该评估，需要分别将候选项代入空缺位置，计算给定候选词时完整句子的条件概率，并选择概率最高的候选项作为模型预测结果。

## Appendix  
### How to download pretrained GPT-2?  
在命令行运行以下指令  
```bash  
pip install -U huggingface-cli  
huggingface-cli download openai-community/gpt-2 --local-dir path/to/pretrained_dir  
```
