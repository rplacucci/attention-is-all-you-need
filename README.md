# Attention Is All You Need

This repository contains my personal implementation of **Attention Is All You Need** in PyTorch. The project covers training on the **WMT 2014 English-to-German** translation benchmark with evaluation using the **BLEU** score against the [official Transformer implementation](https://github.com/tensorflow/tensor2tensor).
## 🌟 Introduction

## 🏗️ Model Overview

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/rplacucci/attention-is-all-you-need.git
cd attention-is-all-you-need
```

**2. Setup the local environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```
## 🚀 Usage

### Training

Build a shared vocabulary:
```bash
python -m vocab.build_vocab --src_lang=en --tgt_lang=de --vocab_size=37000 --min_frequency=2 
```

Calculate the per-example token ID lengths for dynamic batching:
```bash
python -m vocab.survey --src_lang=en --tgt_lang=de
```

Train on the WMT14 English→German benchmark:
```bash
torchrun --standalone --nproc-per-node=4 -m scripts.train --model_config=base --src_lang=en --tgt_lang=de
```

Monitor the training progress via `tensorboard`:
```bash
tensorboard --logdir=logs
```
### Evaluation

Evaluate on the BLEU metric using 🤗 `evaluate`:
```bash
torchrun --standalone --nproc-per-node=1 -m scripts.test --model_config=base --src_lang=en --tgt_lang=de
```

Plot attention maps across several Transformer layers:
```bash
python -m outputs.plot_attn --src_lang=en --tgt_lang=de
```
### Key options

Adjust the training and evaluation routines with the following parameters according to you computational resources:
- `--src_language` source language (default: `en`)
- `--tgt_language` target language (default: `de`)
- `--vocab_size` size of shared token vocabulary (default: `37000`)
- `--min_frequency` minimum number of BPE merges (default: `2`)
- `--nproc-per-node` number of GPUs to train with (default: `4`)
- `--model_config` Transformer architecture (default: `base`)
- `--batch_size` number of samples per validation step (default: `32`)
- `--grad_accum_steps` number of gradient accumulation steps (default: `1`)
- `--logdir` directory where `tensorboard` saves log files (default: `logs`)

All other parameters (e.g., number of training steps, tokens per batch, label smoothing, etc.) have been hard-coded to mimic the original implementation by Vaswani et al.
## 📊 Results

### Training loss
The validation cross-entropy loss exhibits the typical learning behavior for a Transformer model:
- The initial loss (not shown) is approximately $\log V \approx 11$ (with vocabulary size $V=37,000$), which reflects the entropy of random token prediction.
- The model undergoes a steep decline in the first ~10k steps as it rapidly acquires shallow lexical and syntactic regularities.
- By ~30k steps the rate of improvement slows markedly and the loss plateaus near 2.8

![validation loss curve](./outputs/loss_de.png)

**Note:** Pre-training took ~9.5 hours on 4 NVIDIA A40 GPUs.
### BLEU evaluation

The following table summarizes BLEU scores on the **WMT14 English→German** test set for several well-known sequence-to-sequence models, compared against this implementation:

|**Model**                   | **BLEU** |
| :------------------------- | :------: |
| ByteNet                    | 23.8     |
| GNMT + RL                  | 24.6     |
| ConvS2S                    | 25.2     |
| MoE                        | 26.0     |
| GMNT + RL Ensemble         | 26.3     |
| ConvS2S Ensemble           | 26.4     |
| Tranformer (base)          | 27.3     |
| Transforer (big)           | 28.4     |
| *My Transformer (base)*    | *31.2*   |

Overall, this implementation achieves **31.2 sacreBLEU**, which is **3-4 BLEU higher** than the original Transformer-base and competitive with or surpassing the Transformer big baseline. This difference in performance can be explained by a number of reasons:
- The original Transformer results reported in Vaswani et al. (2017) used a slightly smaller preprocessed WMT14 dataset and multi-bleu.perl for evaluation, whereas this implementation uses the 🤗 WMT14 dataset and sacreBLEU, which standardizes tokenization
- While this implementation used the same hyperparameters for preprocessing (e.g., subword tokenization marges) and decoding (e.g., beam size, length penalty, etc.), variations in the dataset could have yielded a gain of several BLEU points
- Modern training infrastructure (using Pytorch v2.8 vs TensorFlow v1.2-1.4) also tends to stabilize optimization and improve results

Overall, while absolute values differ due to these factors, the results agree with the trends reported in Vaswani et al.; moreover, the sacreBLEU score for **GermanEnglish** translation is higher at **33.9**, which is consistent with other implementations.
## Attention Maps

The correctness of this implementation can be further validated by inspecting the **attention maps** and confirming whether they qualitatively align with the findings reported by Vaswani et al. (2017) and the papers that followed. Below is a summary of the key results from this implementation, using the example sentence pair:
```
English: the quick brown fox jumps over the lazy dog
German: der schnelle braune Fuchs springt über den faulen Hund
```
### Encoder self-attention

The encoder layers show the expected progression from local to global context:
- Early layers: Strong diagonal pattern → tokens attend mostly to themselves and close neighbors
- Middle layers: Mixture of local and long-range heads → phrase-level dependencies emerge (e.g., adjective–noun pairs)
- Late layers: Sparser, more specialized heads → capture long-distance dependencies and global semantics

![encoder self-attention](./outputs/attn_maps/en-de/select/encoder_attention_subplots.png)
### Decoder self-attention (masked)

Decoder self-attention is causal, so tokens only attend to previous ones. The maps reveal:
- Early layers: Local context focus → next-word prediction relies on immediate neighbors
- Middle layers: Broader context integration → phrase-level cohesion
- Late layers: Targeted long-range dependencies → attention stays relatively sharp, but specific heads specialize in linking distant yet semantically critical tokens (e.g., subject–verb agreement, clause-level context)

![decoder self-attention](./outputs/attn_maps/en-de/select/decoder_self_attention_subplots.png)
### Decoder cross-attention

Cross-attention aligns target tokens with source tokens. Behavior closely mirrors neural machine translation alignments:
- Early layers: Noisy alignments, but early word-to-word links appear
- Middle layers: Clearer phrase-level alignments
- Late layers: Stable, specialized heads → combine sharp lexical alignment with broad semantic context

![decoder cross-attention](./outputs/attn_maps/en-de/select/decoder_cross_attention_subplots.png)
## 🔮 Future Work

- Implement baseline based on 🤗 `transformers`
- Extend to translation to other languages (e.g., French, Czech, etc.)
## 📚 References

- Vaswani, A., et al. (2017). *Attention is All You Need*.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*.