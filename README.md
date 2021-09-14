<img src="./htm.png" width="500px"></img>

## Hierarchical Transformer Memory (HTM) - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2105.14039">Hierarchical Transformer Memory</a> (HTM) for Pytorch. This Deepmind paper proposes a simple method to allow transformers to attend to memories of the past efficiently. This is largely based on the code <a href="https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory">here</a>.

## Install

```bash
$ pip install htm-pytorch
```

## Usage

```python
import torch
from htm_pytorch import HTMAttention

attn = HTMAttention(
    dim = 512,
    heads = 8,               # number of heads for within-memory attention
    dim_head = 64,           # dimension per head for within-memory attention
    topk_mems = 8,           # how many memory chunks to select for
    mem_chunk_size = 32,     # number of tokens in each memory chunk
    add_pos_enc = True       # whether to add positional encoding to the memories
)

queries = torch.randn(1, 128, 512)     # your queries, in this example, 128 tokens
memories = torch.randn(1, 8888, 512)   # memories, of any size, in this example 8888
mask = torch.ones(1, 8888).bool()      # memory mask

attended = attn(queries, memories, mask = mask) # (1, 128, 512)
```
## Citations

```bibtex
@misc{lampinen2021mental,
    title   = {Towards mental time travel: a hierarchical memory for reinforcement learning agents}, 
    author  = {Andrew Kyle Lampinen and Stephanie C. Y. Chan and Andrea Banino and Felix Hill},
    year    = {2021},
    eprint  = {2105.14039},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
