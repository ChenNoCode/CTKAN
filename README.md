# CTKAN

CTKAN is a medical image segmentation model that injects continuous-thought dynamics into the spline activity space of KAN. It keeps the UKAN-style encoder-decoder backbone and replaces the deepest encoder and decoder token blocks with spline-level dynamic KAN blocks.

## Core Idea

In a standard KAN layer, each spline edge contributes:

```text
sum_k spline_weight[o, i, k] * B_k(x_i)
```

CTKAN builds an internal activity state for each spline coefficient:

```text
z^t[b, o, i, k]
```

The final activity state generates a dynamic gate:

```text
spline_weight[o, i, k] * gate[b, o, i, k] * B_k(x_i)
```

This makes each spline coefficient input-adaptive while preserving the original segmentation pipeline.

## Project Layout

```text
CTKAN/
  arch/
    __init__.py
    common.py
    kan.py
    ctkan.py
  inputs/      # copy datasets here
  outputs/     # training outputs are written here
  train.py
  val.py
  dataset.py
  losses.py
  metrics.py
```

## Training

Dataset folders follow the existing project format:

```text
inputs/{dataset}/images
inputs/{dataset}/masks/0
```

Example:

```bash
python train.py --arch CTKAN --dataset busi --input_w 256 --input_h 256 --name CTKAN --epochs 400 --batch_size 4 --data_dir ./inputs
python val.py --name busi/CTKAN
```

## CTKAN Parameters

| Argument | Default | Description |
|---|---:|---|
| `--ctm_ticks` | 20 | Internal thought ticks for spline activity updates |
| `--ctm_dhidden` | 64 | Hidden width of synchronization feedback projections |
| `--ctm_dropout` | 0.2 | Dropout in synchronization feedback projections |
| `--ctm_daction` | 1024 | Sampled activity pairs for recurrent feedback |
| `--ctm_dout` | 1024 | Sampled activity pairs for output gate feedback |
| `--ctm_nself` | 32 | Number of self-pairs in synchronization sampling |
| `--ctm_memory` | 10 | EMA memory length for internal spline activity |
| `--ctm_scale_init` | 1e-3 | Initial residual scale of the dynamic spline gate |

## Suggested Ablations

- Sweep `--ctm_ticks` over values such as `1, 5, 10, 20`.
- Sweep `--ctm_memory` to test short versus long internal activity traces.
- Sweep `--ctm_daction` and `--ctm_dout` to measure synchronization sampling width.


