# 1.3 Beyond U-Net: mask transformers, foundation features, and the tools around them

Sub-step 1.3 of stage 1. Scope: the newer side of the model ladder. Mask2Former and EoMT represent "mask classification", SegFormer the light-decoder idea, DINOv3 the frozen foundation backbone our stage 6 uses, and MONAI the reference implementations of medical losses and metrics.

Pinned commits:
- tue-mps/eomt: [`aa017c3`](https://github.com/tue-mps/eomt/tree/aa017c34afbe023de9462b17ae6067b52fef9867)
- facebookresearch/dinov3: [`6876159`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)
- facebookresearch/Mask2Former: [`9b0651c`](https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a)
- NVlabs/SegFormer: [`65fa8cf`](https://github.com/NVlabs/SegFormer/tree/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32)
- Project-MONAI/MONAI: [`1dba735`](https://github.com/Project-MONAI/MONAI/tree/1dba7356266e39d414bf76dcd75881184ba64bf9)

## Who is behind it

| Work | People and affiliation as printed | Source |
|---|---|---|
| Mask2Former (CVPR 2022) | Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar; Facebook AI Research and University of Illinois Urbana-Champaign (Cheng's work done as a FAIR intern) | [arXiv:2112.01527](https://arxiv.org/abs/2112.01527) |
| SegFormer (NeurIPS 2021) | Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo; The University of Hong Kong, Nanjing University, NVIDIA, Caltech | [arXiv:2105.15203](https://arxiv.org/abs/2105.15203) |
| EoMT (CVPR 2025, highlight) | Tommie Kerssies, Narges Norouzi, Gijs Dubbelman (Eindhoven University of Technology); Daan de Geus (TU/e and RWTH Aachen); Alexander Hermans, Bastian Leibe (RWTH Aachen); Niccolò Cavagnero, Giuseppe Averta (Polytechnic of Turin) | [arXiv:2503.19108](https://arxiv.org/abs/2503.19108) |
| DINOv3 (2025) | Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose and others; Meta AI Research, with WRI and Inria | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) |
| MONAI (2022 paper) | M. Jorge Cardoso, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot and 52 others; "initially led by NVIDIA and King's College London", now a consortium of 16 universities and industrial partners | [arXiv:2211.02701](https://arxiv.org/abs/2211.02701) |

## Three ways to produce a mask

Everything in this project falls into one of three designs. Knowing which is which is the most useful thing to take from this sub-step.

| Design | How it predicts | Examples | Our ladder |
|---|---|---|---|
| Per-pixel classification with a decoder | Encoder extracts features, decoder upsamples them, every pixel gets a class score | U-Net, SegFormer, PraNet | Rungs 1-2 (U-Net from scratch, U-Net with ResNet-34) |
| Mask classification | N learned queries; each predicts one mask and one class label; the final map combines them | Mask2Former, EoMT | Server stage only |
| Frozen backbone + light head | A pretrained backbone is not trained at all; a tiny head maps its patch features to classes | DINOv3 linear probe | Rung 3 |

Mask classification exists because per-pixel classification struggles with instances: it can say "these pixels are polyp" but not "these are two different polyps". For our binary task, per-pixel is enough. Mask classification is in the project because it is the current state of the art on general benchmarks, and the server stage tests whether that carries over.

## Mask2Former: masked attention and a cheaper loss

Three ideas, each visible in code or config:

1. **Masked attention.** Each query attends only to pixels inside the mask it predicted at the previous decoder layer, instead of the whole image (paper, Section 3.2.1). Attention stays local to the object being refined.
2. **One resolution per decoder layer.** Rather than feeding high-resolution features everywhere, the decoder cycles through a feature pyramid one scale per layer (Section 3.2.2).
3. **Loss on sampled points, not all pixels.** The mask loss is computed at 12,544 points per mask. They are chosen by oversampling 3× and keeping 75% of them at the most uncertain locations, with the rest random ([`MF criterion.py#L163-L185`](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/criterion.py#L163-L185); values in [`MF configs/.../maskformer2_R50_bs16_160k.yaml#L36-L38`](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml#L36-L38)). The paper reports this cuts training memory per image from 18 GB to 6 GB without hurting accuracy.

Loss weights in the same config: class 2.0, mask (BCE) 5.0, Dice 5.0, no-object class 0.1, 100 queries, 9 decoder layers (L22-L35). The no-object weight is applied in the criterion at [`MF criterion.py#L114`](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/criterion.py#L114).

Published result (paper, Table 3, ADE20K, single-scale): 47.2 mIoU with ResNet-50 at 512×512; 56.1 mIoU with Swin-L at 640×640.

**Why it matters for us:** point-sampled loss is the same instinct as PraNet's boundary weights and the U-Net weight map. Spend the loss budget where the mask is uncertain.

## SegFormer: a decoder that is almost nothing

The SegFormer decoder is four linear layers, an upsample, a concatenation and a 1×1 fusion ([`SF segformer_head.py#L50-L82`](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/mmseg/models/decode_heads/segformer_head.py#L50-L82)):

1. Project each of the 4 encoder outputs to the same width with an MLP (`linear_c1` … `linear_c4`).
2. Resize all of them to the 1/4-resolution grid with bilinear interpolation (L72).
3. Concatenate and fuse with one convolution (L82), then a 1×1 classifier (L62).

The paper's argument: the transformer encoder already has a large effective receptive field, so the decoder does not need to add context, only combine scales. The encoder also drops positional encodings and uses a 3×3 convolution inside the feed-forward layer instead, so it works at any input size.

Published (paper, Tables 1-2, ADE20K): SegFormer-B0 37.4 mIoU with 3.8M parameters; SegFormer-B5 51.0 single-scale / 51.8 multi-scale with 84.7M.

**Why it matters for us:** it is the argument for rung 3. If the backbone features are good enough, a very small head should be enough too. Our linear head on DINOv3 is the extreme version of that bet.

## EoMT: a plain ViT as the whole segmentation model

EoMT asks whether the adapter, pixel decoder and transformer decoder that Mask2Former needs on top of a ViT are still necessary when the ViT is large and well pretrained. It removes all three.

How the code does it:

| Step | What happens | Code |
|---|---|---|
| Queries enter late | 100 learned query tokens are concatenated to the patch tokens only before the last 4 ViT blocks | [`EO models/eomt.py#L166-L169`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L166-L169), `num_blocks=4` at [L24](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L24), queries at [L35](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L35) |
| Masks from a dot product | Each query's embedding is dot-multiplied with upscaled patch features to give a mask; a linear layer gives its class | [`EO models/eomt.py#L55-L70`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L55-L70) |
| Masked attention, ViT style | In those last blocks, a query may only attend to patches where its current mask logit is > 0 | [`EO models/eomt.py#L121-L141`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L121-L141) |
| Mask annealing | For each block, a probability decides whether the mask is applied per query; it decays from 1 to 0 on a poly schedule, block by block | [`EO models/eomt.py#L72-L82`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py#L72-L82), [`EO training/lightning_module.py#L199-L209`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/training/lightning_module.py#L199-L209) |
| Schedule | Annealing windows for the 4 blocks: steps 0-3000, 6520-9780, 9780-13040, 13040-16300 (ADE20K, ViT-L) | [`EO configs/dinov3/ade20k/semantic/eomt_large_512.yaml#L12-L14`](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/configs/dinov3/ade20k/semantic/eomt_large_512.yaml#L12-L14) |

Mask annealing is the clever part. Masked attention helps training, but computing intermediate masks at inference is expensive. So training starts with masking and gradually removes it, until the model no longer needs it. At inference there is no masking at all.

Published results:

| Comparison | Quality | Speed | Source |
|---|---|---|---|
| ADE20K semantic, ViT-L, DINOv2, 512×512 | EoMT 58.4 vs ViT-Adapter + Mask2Former 58.9 mIoU | 92 vs 21 FPS | Paper, Table 5 |
| COCO panoptic, ViT-L | EoMT 56.0 vs 57.1 PQ | 128 vs 29 FPS | Paper, Section 4.2 |
| ADE20K semantic, ViT-L, DINOv3 | 59.5 mIoU | 137 / 200 FPS on H100 (default / max-autotune compile) | [EoMT DINOv3 model zoo](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/model_zoo/dinov3.md) |

**The caveat that decides our plan:** the gap grows as models get smaller or pretraining gets weaker. The paper reports EoMT 5.8 PQ behind the full Mask2Former pipeline with ViT-S, against 0.7 PQ with ViT-g (Table 3). With ViT-L and ImageNet-1K pretraining the gap is 6.1 PQ, against 1.1 with DINOv2 (Table 2). A ViT-S EoMT on the Air would test exactly the regime where EoMT is weakest. That is why EoMT stays in the server stage.

## DINOv3: the backbone for rung 3

**What it is.** A family of self-supervised vision models from Meta, trained on LVD-1689M, a curated set of 1,689 million images drawn from about 17 billion public Instagram images (paper). The large models are distilled into smaller ones; the ViT-S/16 we will use has 21M parameters (README model table).

**Gram anchoring**, the paper's key idea: during long training, global features keep improving while dense patch features degrade, which hurts segmentation. Gram anchoring adds a loss that keeps the student's patch-to-patch similarity matrix (the Gram matrix) close to that of an earlier checkpoint. It constrains the similarity structure, not the features themselves.

**Published dense quality** (paper, Table 14, ADE20K mIoU, models of comparable size):

| Size | DINOv3 | DINOv2 | SigLIP 2 | PE-core |
|---|---|---|---|---|
| S | 47.0 | 45.5 | – | – |
| S+ | 48.8 | – | – | – |
| B | 51.8 | 48.4 | 41.6 | 37.4 |
| L | 54.9 | 48.8 | 43.6 | 39.7 |

The 7B teacher reaches 55.9 mIoU with a linear classifier (paper, Section 6.1.2). Two things stand out. Text-aligned models (SigLIP 2, PE-core) have much weaker dense features than DINO models of the same size. And DINOv3 ViT-S beats DINOv2 ViT-S by 1.5 points, which sets the size of the difference to expect if we fall back to DINOv2.

**The official linear head** is BatchNorm, dropout and a single 1×1 convolution on patch features, followed by bilinear upsampling to the output size ([`DV linear_head.py#L29-L31`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/models/heads/linear_head.py#L29-L31), [`#L97`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/models/heads/linear_head.py#L97)). The ADE20K recipe: 40,000 iterations, lr 1e-3, weight decay 1e-3, cross-entropy only (Dice weight 0), features from the last layer, 512 px, sliding-window evaluation ([`DV config-ade20k-linear-training.yaml#L8-L49`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml#L8-L49)).

**The closest official example to our task** is the foreground-segmentation notebook ([`DV notebooks/foreground_segmentation.ipynb`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/foreground_segmentation.ipynb)):
- 9 training images at 768 px with ViT-L.
- Each mask is reduced to the 16×16-pixel patch grid with a box filter, giving a soft "fraction of foreground" label per patch.
- A logistic regression is trained on patch features. The regularisation strength C is chosen from 8 values by leave-one-image-out validation, because patches from the same image are strongly correlated.
- A 3×3 median filter smooths the predicted patch map.

The leave-one-image-out detail matches the protocol theme of 1.1: never validate on data correlated with training data.

**Licence.** Weights are released under the DINOv3 License and require accepting it on request (README, "Pretrained models" and "License"). You have to accept it in your own account; I cannot do that for you.

## MONAI: reference implementations we will test against

- `DiceLoss` computes (2·TP + smooth_nr) / (2·(TP+FP+FN) + smooth_dr), with both smoothing terms 1e-5 by default ([`MO monai/losses/dice.py#L210-L211`](https://github.com/Project-MONAI/MONAI/blob/1dba7356266e39d414bf76dcd75881184ba64bf9/monai/losses/dice.py#L210-L211), defaults at [L66-L68](https://github.com/Project-MONAI/MONAI/blob/1dba7356266e39d414bf76dcd75881184ba64bf9/monai/losses/dice.py#L66-L68)). The `batch` flag chooses between Dice per image and Dice over the whole batch, the same distinction as SMP's reductions in 1.2.
- `DiceCELoss` is Dice + cross-entropy with weights `lambda_dice` and `lambda_ce`, both 1.0 ([`MO monai/losses/dice.py#L718-L743`](https://github.com/Project-MONAI/MONAI/blob/1dba7356266e39d414bf76dcd75881184ba64bf9/monai/losses/dice.py#L718-L743)), the same default as nnU-Net.
- `SurfaceDiceMetric` (normalised surface Dice, a boundary metric) takes a tolerance per class. Its docstring warns that by default its boundary computation differs from DeepMind's reference implementation ([`MO monai/metrics/surface_dice.py#L32-L79`](https://github.com/Project-MONAI/MONAI/blob/1dba7356266e39d414bf76dcd75881184ba64bf9/monai/metrics/surface_dice.py#L32-L79)).

We will not train with MONAI. We will use it, and SMP, as a second opinion: our own metric functions must agree with theirs on random inputs before any result is trusted (stage 2).

## What this means for rung 3 (stage 6)

| Question | What the sources say | Our decision |
|---|---|---|
| Input resolution | Patch size is 16, so the head sees a (H/16)×(W/16) grid. At 256 px that is 16×16 patches for the whole image | Run 256 and 448 px (28×28 patches) and report Boundary IoU separately, since the coarse grid should hurt edges most |
| Head | Official: BN + 1×1 conv + bilinear upsampling; notebook: logistic regression | Official linear head (trainable in PyTorch, same loss and evaluation as other rungs); logistic regression as a sanity check |
| Which layer | Official config uses the last layer | Last layer first; concatenating the last 4 layers as an ablation |
| Training cost | Backbone frozen | Extract features once, cache them to disk, train only the head; report extraction time separately |
| Domain gap | DINOv3 is trained on web photos, not endoscopy | Frozen features may transfer poorly to colonoscopy. If so, that is a finding, not a failure, and it motivates the server-stage fine-tuning |
| Licence fallback | DINOv2 ViT-S is 1.5 mIoU behind on ADE20K | If DINOv3 access is not granted, DINOv2 ViT-S, labelled in every table |

## What we take, what we change

| Take | From |
|---|---|
| The three-design framing for the README and the site | this sub-step |
| Official linear-head design and feature caching for rung 3 | DINOv3 |
| Leave-one-image-out thinking: validation must not be correlated with training | DINOv3 notebook |
| Uncertainty-focused loss as a shared idea across U-Net, PraNet and Mask2Former, for the site's "loss" page | Mask2Former |
| Quality-and-speed reporting, FPS next to mIoU | EoMT |
| MONAI and SMP as reference implementations to test our metrics against | MONAI, SMP |

| Not taken now, and why |
|---|
| EoMT and Mask2Former on the Air: the regime that fits in 16 GB (ViT-S) is where EoMT trails most; both go to the server stage |
| SegFormer as a separate rung: it would add a fourth model without answering a new question; mentioned as context only |

## Questions you should be able to answer

**What is the difference between per-pixel classification and mask classification?**
Per-pixel gives every pixel a class score. Mask classification predicts a set of masks, each with one class, and combines them. Per-pixel cannot separate two touching objects of the same class; mask classification can, which is why one model handles semantic, instance and panoptic segmentation.

**What problem does mask annealing solve?**
Masked attention improves training, but computing intermediate masks at inference costs time. Annealing drops the masks gradually during training, so the final model is accurate without them and runs as a plain ViT.

**Why use a frozen backbone with a linear head?**
It measures how good the features are on their own. It is cheap (minutes of training after one feature-extraction pass) and hard to overfit with little data. If it matches a fully trained U-Net, the backbone has done most of the work.

**Why might DINOv3 do worse on polyps than on ADE20K?**
ADE20K is everyday scenes, close to the web photos DINOv3 was trained on. Colonoscopy images look nothing like that. Frozen features may not separate polyp from tissue well, and the 16-pixel patch grid limits boundary precision.

**Why do text-image models like SigLIP 2 score lower on segmentation than DINO models?**
Contrastive text-image training rewards a good global summary of the image. It does not reward precise patch-level features. Self-supervised DINO training does, and Gram anchoring specifically protects dense quality.

**Why does Mask2Former compute its loss on sampled points?**
Full-resolution masks for 100 queries use a lot of memory. Sampling about 12k points, focused on uncertain regions, gives the same accuracy at a third of the memory per image.

## Glossary

- **Query (object query)**: a learned vector that, through attention, comes to represent one object or region and predicts its mask.
- **Masked attention**: attention restricted to a region, here the query's current predicted mask.
- **Mask annealing**: gradually turning masked attention off during training.
- **Frozen backbone**: a pretrained network whose weights are not updated; only the head on top is trained.
- **Linear probe**: a single linear layer trained on frozen features, used to measure feature quality.
- **Patch size**: a ViT splits the image into squares of this many pixels (16 for DINOv3); the output grid is image size ÷ patch size.
- **Gram matrix**: the matrix of similarities between all pairs of patch features in an image.
- **Effective receptive field**: the part of the input that actually influences one output location.
- **Normalised surface Dice (NSD)**: the share of boundary points that lie within a set distance of the true boundary.
