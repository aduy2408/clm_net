# CLM-Net

CLM-Net is the implementation for **"CLM-Net: A Channel-Optimized Lightweight Hybrid Network for Varroa Mite Segmentation"**, accepted at **MIWAI 2026**.

The model is a lightweight LM-Net variant for small-object Varroa mite segmentation. It keeps the encoder-decoder backbone and replaces the expensive full-channel attention components with:

- **LNAB**: Lightweight Neighborhood Attention Block. It applies neighborhood attention to only part of the skip-feature channels and keeps the rest through an identity branch.
- **LGB**: Lightweight Global Bottleneck. It compresses pyramid-pooled bottleneck features with a 1x1 projection and depthwise convolution before global self-attention.

From the paper, **CLM-Net-192** is the preferred configuration: test Dice `70.32%`, test IoU `54.24%`, `1.71M` parameters, and `8.005` GFLOPs.

## Structure

```text
clm_net/
  model/
    modules.py          # LNAB, LGB, skip fusion, pyramid pooling, reparam conv, upsampling
    CLM_Net.py          # CLM_Net model definition only
  utils/
    train_clm_net_varroa.py
```

## Requirements


```bash
pip install -r requirements.txt
```

## Dataset Layout

`train_clm_net_varroa.py` expects each split to contain `videos/` and `labels/`:

```text
DATA_ROOT/
  train/
    videos/
    labels/
  val/
    videos/
    labels/
  test/
    videos/
    labels/
```

Labels are bounding-box text files. The runner converts boxes to rectangular binary masks during loading.

## Train

Default CLM-Net-192:

```bash
python clm_net/utils/train_clm_net_varroa.py \
  --root /path/to/varroa_data \
  --seed 42 \
  --lgb-bottleneck 192 \
  --lnab-kind partial \
  --lnab-ratios 0.5 0.5 0.5 0.5 \
  --batch-size 8 \
  --epochs 100 \
  --patience 10 \
  --amp
```

Useful variants:

```bash
# CLM-Net-128
python clm_net/utils/train_clm_net_varroa.py --root /path/to/varroa_data --lgb-bottleneck 128

# CLM-Net-384
python clm_net/utils/train_clm_net_varroa.py --root /path/to/varroa_data --lgb-bottleneck 384

# LNAB-only ablation: disable LGB compression by using a wider bottleneck if needed
python clm_net/utils/train_clm_net_varroa.py --root /path/to/varroa_data --lgb-bottleneck 384

# Disable LNAB for ablation
python clm_net/utils/train_clm_net_varroa.py --root /path/to/varroa_data --lnab-kind identity
```

Outputs are written under:

```text
runs/clm_net_varroa/seed_<seed>/
  best.pt
  config.json
  manifest.json
  clm_net_varroa_<timestamp>.csv
  evaluation_metrics.json
```

## Main Flags

- `--lgb-bottleneck`: LGB compressed channel width. Paper variants use `64`, `128`, `192`, `256`, and `384`.
- `--lnab-kind`: `partial` for CLM-Net, `identity` for ablation.
- `--lnab-ratios`: attended channel ratios for the four skip stages. Paper setting is `0.5 0.5 0.5 0.5`.
- `--filters`: channel list. Default is `[24, 24, 48, 96, 192]`.
- `--upsample-kind`: `bilinear_conv` by default.
- `--se-kind`: `sse` by default.

Backward-compatible aliases are kept for older scripts:

- `--gft-bottleneck` maps to `--lgb-bottleneck`
- `--partial-ratios` maps to `--lnab-ratios`
- `--skip-attention` maps to `--lnab-kind`

## Citation

```bibtex
@inproceedings{le2026clmnet,
  title     = {CLM-Net: A Channel-Optimized Lightweight Hybrid Network for Varroa Mite Segmentation},
  author    = {Le, Ngoc-Duy and Hoang, Trung-Nguyen and Tran, Ly-Minh-Hoang and Phan, Thi-Thu-Hong},
  booktitle = {Proceedings of MIWAI 2026},
  year      = {2026}
}
```
