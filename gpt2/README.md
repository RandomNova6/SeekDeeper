[📖中文 ReadMe](./README_zh.md)
## Introduction

In this GPT-2 implementation, we demonstrate how to pre-train on the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset, load the official pre-trained weights from [Hugging Face](https://huggingface.co/openai-community/gpt2/tree/main) into our model, and directly evaluate on the [Children's Book Test (CBT)](https://arxiv.org/pdf/1511.02301) dataset without fine-tuning, thereby reproducing the results reported in the paper.

## Model details

### Key differences with GPT-1
1. GPT-2 moved `LayerNorm` to the input of each sub-block and added an extra `LayerNorm` at the end. For analysis on `Pre-LN`, refer to the paper [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
2. GPT-2 expanded the vocabulary size from 40,478 to 50,257 and increased the context length from 512 to 1,024.
3. GPT-2 used a modified initialization method, considering the accumulation of residual paths as the model depth increases. During initialization, the weights of the residual mapping layer are scaled by $1/\sqrt{N}$, where $N$ is the number of residual layers.

### [Byte-pair encoding (BPE)](./modules/bpe.py)

BPE is a tokenization method that builds larger subword units by merging the most frequently occurring pairs of characters, reducing the vocabulary size and addressing the issue of rare words. It requires training on a corpus to obtain the vocabulary before encoding and decoding.

Our implementation is largely based on Karpathy's [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py). For further details on BPE training, Karpathy's [minbpe](https://github.com/karpathy/minbpe) provides a useful reference.

The BPE used in GPT-2 has some improvements over the version used in GPT-1. Unlike GPT-1, which was based on Unicode characters, GPT-2's BPE operates at the byte level. This means GPT-2 can handle various character sets and special symbols more flexibly, especially non-ASCII characters and emojis, which is particularly helpful for multilingual support and processing non-English text.

## [Pre-training](./pretrain.ipynb)

The general steps are similar to GPT-1's pre-training.

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Sec. 2.1 provides a detailed description of how OpenWebText was generated, but does not mention much about the training hyperparameters in the subsequent paragraphs.

## [Inference](./inference.ipynb)

GPT-2 substantially improves text generation capability compared with GPT-1. We reproduce the evaluation on the CN subset of the Children's Book Test (CBT) reported in the paper. This subset requires the model to select an appropriate noun from ten candidates to fill a blank in a contextual paragraph, and prediction accuracy is used as the evaluation metric. To perform this evaluation, each candidate is substituted into the blank, the conditional probability of the full sentence is computed given the candidate word, and the candidate with the highest probability is selected as the model prediction.

## Appendix

### How to download pretrained GPT-2?

Run the following commands in the terminal:

```bash
pip install -U huggingface-cli
huggingface-cli download openai-community/gpt-2 --local-dir path/to/pretrained_dir
```
