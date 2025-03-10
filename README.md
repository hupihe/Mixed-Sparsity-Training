
# Mixed Sparsity Training

This a repository for training GPTs with the Mixed Sparsity Training (MST) proposed in xx["Mixed Sparsity Training: Achieving 4x FLOP Reduction for Transformer Pretraining"](https://arxiv.org/abs/2408.11746).  We use the [nanoGPT](https://github.com/karpathy/minGPT) code base. NanoGPT is a lightweight version of the GPT-2 model trained on the OpenWebText dataset. Our experiment implementation is derived from the small GPT-2 model. The architecture comprises 12 transformer layers and 12 attention heads, with an embedding size set to 768. The text is tokenized with the GPT-2 tokenizer. We adopt the train-validation split provided by nanoGPT. The training set comprises 9 billion tokens, and the validation set contains 4.4 million tokens. During training, we optimize the cross-entropy loss for next-token prediction. Consistent with nanoGPT, we employ GELU activations while disabling bias and Dropout. Distributed data parallelism with gradient accumulation is employed to enable a batch size of 480. Training is conducted with bfloat16 precision on machines with 4 A100 GPUs. 

## install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## quick start

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```
$ torchrun --nproc_per_node=4 sparse_train.py config/mst.py
```
