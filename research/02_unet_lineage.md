# 1.2 The U-Net lineage: the paper, three implementations, and what "Revisited" found

Sub-step 1.2 of stage 1. Scope: the original U-Net and the three implementations our stages 4-5 build on (milesial/Pytorch-UNet, segmentation_models.pytorch, nnU-Net), plus the nnU-Net Revisited paper, which sets the standard for how we compare models.

Pinned commits:
- milesial/Pytorch-UNet: [`21d7850`](https://github.com/milesial/Pytorch-UNet/tree/21d7850f2af30a9695bbeea75f3136aa538cfc4a)
- qubvel-org/segmentation_models.pytorch (SMP): [`5b8e53f`](https://github.com/qubvel-org/segmentation_models.pytorch/tree/5b8e53f0ebece38823477b656c442ea389b71669)
- MIC-DKFZ/nnUNet: [`90cc65a`](https://github.com/MIC-DKFZ/nnUNet/tree/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e)

`ML`, `SMP`, `NN` below are shorthand for those three repos at the pinned commits; every reference is a real link.

## Who is behind it

| Work | People and affiliation | Source |
|---|---|---|
| U-Net (MICCAI 2015) | Olaf Ronneberger (Computer Science Department and BIOSS Centre, University of Freiburg), Philipp Fischer, Thomas Brox | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| nnU-Net (Nature Methods 2021) | Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein (German Cancer Research Center (DKFZ), Heidelberg University, Heidelberg University Hospital; Kohl also DeepMind) | [arXiv:1904.08128](https://arxiv.org/abs/1904.08128) |
| nnU-Net Revisited (MICCAI 2024) | Fabian Isensee, Tassilo Wald, Constantin Ulrich, Michael Baumgartner, Saikat Roy, Klaus Maier-Hein, Paul Jäger (DKFZ Division of Medical Image Computing, Helmholtz Imaging, Heidelberg University, NCT Heidelberg) | [arXiv:2404.09556](https://arxiv.org/abs/2404.09556) |
| milesial/Pytorch-UNet | maintainer `milesial`; GitHub profile lists NVIDIA | [profile](https://github.com/milesial) |
| segmentation_models.pytorch | created by Pavel Iakubovskii (`qubvel`); GitHub profile: AI Engineer at Praktika.ai, formerly Hugging Face | [profile](https://github.com/qubvel) |

nnU-Net is maintained mainly by Fabian Isensee (1,726 commits; the next contributor has 30, per the GitHub contributors API).

## The original U-Net (2015)

What the paper actually specifies (arXiv:1505.04597, Sections 2-3):

| Part | Paper |
|---|---|
| Encoder | Repeated two 3×3 convolutions, **unpadded**, each followed by ReLU, then 2×2 max pooling; channels double at each step |
| Decoder | Upsampling plus a 2×2 "up-convolution" that halves channels, concatenation with the **cropped** encoder features, two 3×3 convolutions |
| Output | 1×1 convolution from a 64-channel feature vector; 23 convolutional layers in total |
| Normalisation | None (batch normalisation is not mentioned) |
| Loss | Pixel-wise softmax + cross-entropy with a weight map: w(x) = w_c(x) + w₀·exp(−(d₁+d₂)²/2σ²), w₀ = 10, σ ≈ 5 px, which upweights thin gaps between touching cells |
| Optimizer | SGD (Caffe), momentum 0.99, batch of a single image to fit large tiles |
| Init | Gaussian, std √(2/N) (He initialisation) |
| Augmentation | Elastic deformations: random displacements on a coarse 3×3 grid, std 10 px, bicubic interpolation; dropout at the end of the encoder |
| Claim | Won the ISBI cell tracking challenge 2015 in its categories; a 512×512 image in under a second on a GPU |

Two ideas survived everything that came after: the **skip connections** that carry fine spatial detail from encoder to decoder, and the **weighted loss** that puts emphasis where segmentation is hard. PraNet's boundary-weighted structure loss (sub-step 1.1) is the same idea in a different form.

## How the three implementations differ from the paper

| | Paper 2015 | milesial | SMP `Unet` | nnU-Net (2D) |
|---|---|---|---|---|
| Convolutions | unpadded, output smaller than input | padded (`padding=1`) | padded | padded |
| Normalisation | none | BatchNorm | BatchNorm by default | InstanceNorm + LeakyReLU |
| Encoder | trained from scratch | from scratch, 64→1024 channels | pretrained (ImageNet by default), any of 500+ backbones | from scratch, depth and width set by rules from the data |
| Upsampling | up-convolution | transposed conv, or bilinear if `--bilinear` | nearest-neighbour interpolation | transposed conv |
| Loss | weighted cross-entropy | BCE + Dice | user's choice | Dice + CE, with deep supervision |
| Optimizer | SGD, momentum 0.99 | RMSprop, lr 1e-5, momentum 0.999 | user's choice | SGD Nesterov, momentum 0.99, lr 1e-2, poly decay |
| Validation | not described | random 10%, seed 0 | user's choice | 5-fold cross-validation, seeded |

nnU-Net builds its networks with a separate package; the transposed convolutions are created in [`dynamic-network-architectures building_blocks/unet_decoder.py#L55-L73`](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/7bf2053613e2ec3e76ccac8a5fdc51dd72668287/dynamic_network_architectures/building_blocks/unet_decoder.py#L55-L73).

### milesial/Pytorch-UNet: a clean teaching implementation

- `DoubleConv` is two 3×3 convolutions with `padding=1`, BatchNorm and ReLU ([`ML unet/unet_parts.py#L8-L25`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_parts.py#L8-L25)). Padding keeps the output the same size as the input, so the paper's cropping is not needed. Batch norm makes training stable at the cost of a dependence on batch statistics.
- `Up` pads the upsampled map when its size is off by one, then concatenates ([`ML unet/unet_parts.py#L56-L68`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_parts.py#L56-L68)). This handles inputs whose size is not divisible by 16.
- With `bilinear=True` the deepest level has 512 instead of 1,024 channels (`factor = 2`), so the model is noticeably smaller ([`ML unet/unet_model.py#L13-L23`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_model.py#L13-L23)).
- Loss: BCE plus soft Dice ([`ML train.py#L103-L104`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/train.py#L103-L104)). Optimizer: RMSprop with momentum 0.999 and `ReduceLROnPlateau` on validation Dice ([`ML train.py#L77-L79`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/train.py#L77-L79)). Momentum 0.999 is unusually high for RMSprop.
- Validation: a random 10% split with a fixed seed ([`ML train.py#L48-L50`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/train.py#L48-L50)). Better than no validation at all, which is what PraNet does.
- Published result: Dice 0.988 on over 100k Carvana test images, trained from scratch on 5k images (README). Carvana (car silhouettes on studio backgrounds) is far easier than polyps.

**Two evaluation details that change numbers:**
1. The validation loader uses `drop_last=True` ([`ML train.py#L55`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/train.py#L55)). If the validation set size is not a multiple of the batch size, the last few images are never evaluated.
2. Dice is computed per batch and then averaged over batches ([`ML evaluate.py#L28-L30`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/evaluate.py#L28-L30), [`#L40`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/evaluate.py#L40)). The result depends on the batch size. A per-image mean, which is what PraNet reports, does not.

**A bug found in stage 4.1.** After a CUDA out-of-memory error, `train.py` calls `model.use_checkpointing()` to retry with less memory ([`ML train.py#L223-L228`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/train.py#L223-L228)). That method wraps each block in `torch.utils.checkpoint(...)`, which is a module, not a function ([`ML unet/unet_model.py#L38-L48`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_model.py#L38-L48)), so the retry crashes with `TypeError: 'module' object is not callable`. `tests/test_unet.py` pins this.

One good detail: when both prediction and target are empty, `dice_coeff` returns 1 instead of 0/0 ([`ML utils/dice_score.py#L12-L16`](https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/utils/dice_score.py#L12-L16)). Every Dice implementation has to decide this case, and many don't say how.

### segmentation_models.pytorch: the strong, cheap baseline

- `Unet` defaults: encoder depth 5, ImageNet weights, BatchNorm in the decoder, decoder channels (256, 128, 64, 32, 16), nearest-neighbour upsampling ([`SMP decoders/unet/model.py#L111-L116`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/5b8e53f0ebece38823477b656c442ea389b71669/segmentation_models_pytorch/decoders/unet/model.py#L111-L116)). The decoder is much narrower than milesial's (16-256 channels instead of 64-512), because the pretrained encoder does most of the work.
- Decoder block: interpolate, concatenate the skip, two convolutions ([`SMP decoders/unet/decoder.py#L50-L56`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/5b8e53f0ebece38823477b656c442ea389b71669/segmentation_models_pytorch/decoders/unet/decoder.py#L50-L56)).
- `DiceLoss` sets the loss to zero for images whose target mask is empty ([`SMP losses/dice.py#L134-L135`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/5b8e53f0ebece38823477b656c442ea389b71669/segmentation_models_pytorch/losses/dice.py#L134-L135)). Every polyp image has a polyp, so this does not affect us, but it matters for datasets with negative images.
- Metrics have explicit reduction modes: `micro` pools all pixels of all images; `micro-imagewise` computes the metric per image and averages ([`SMP metrics/functional.py#L275-L311`](https://github.com/qubvel-org/segmentation_models.pytorch/blob/5b8e53f0ebece38823477b656c442ea389b71669/segmentation_models_pytorch/metrics/functional.py#L275-L311)). Pooled Dice is dominated by large polyps; per-image Dice gives a small missed polyp the same weight as a large one. We report per-image, to match the papers, and say so.

### nnU-Net: the recipe that is hard to beat

nnU-Net does not propose a new architecture. It fixes the training recipe and sets everything else by rules derived from the dataset (paper, Results and Methods):

- **Fixed (blueprint):** U-Net template, instance norm and leaky ReLU, loss, schedule, augmentation.
- **Rule-based (inferred):** topology, patch size, batch size, normalisation, target spacing.
- **Empirical:** choice between 2D, 3D and cascade, and post-processing, chosen by cross-validation.

The paper reports a new state of the art on 29 of 49 tasks across 19 datasets, without manual tuning.

The fixed recipe in code:

| Setting | Value | Code |
|---|---|---|
| Learning rate, weight decay, epochs | 1e-2, 3e-5, 1000 epochs × 250 iterations | [`NN nnUNetTrainer.py#L173-L181`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L173-L181) |
| Foreground oversampling | 33% of patches forced to contain foreground | same, L175 |
| Optimizer | SGD, momentum 0.99, Nesterov | [`#L590-L592`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L590-L592) |
| Schedule | lr × (1 − step/max)^0.9 | [`NN polylr.py#L18`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/lr_scheduler/polylr.py#L18) |
| Loss | Dice + cross-entropy, weight 1 each (Dice + BCE in region mode) | [`#L474-L482`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L474-L482) |
| Deep supervision | weights 1, ½, ¼, … per lower resolution; the two lowest outputs are dropped; weights normalised to sum to 1 | [`#L491-L505`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L491-L505) |
| Augmentation | rotation and scaling 0.7-1.4 (p = 0.2 each), Gaussian noise, blur, brightness, contrast, simulated low resolution, gamma, mirroring | [`#L797-L892`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L797-L892) |
| Split | 5-fold cross-validation, seed 12345, written to `splits_final.json` and reused | [`#L668-L670`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L668-L670) |

The split file is the habit to copy: the split is created once, saved, and every later run reads the same file.

A code comment worth knowing for interviews: with DDP, a weight of exactly 0 on an output crashes training with "unused parameters", so nnU-Net uses 1e-6 instead ([`#L494-L500`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L494-L500)).

## nnU-Net Revisited: why our project is a benchmark, not a new model

The paper (arXiv:2404.09556) re-ran many recent "better than U-Net" methods under one protocol. Its conclusion: most claimed gains disappear. It names four pitfalls:

| Pitfall | What it means | How our project avoids it |
|---|---|---|
| P1. Confounded boosters | Gains credited to a new idea actually come from a better encoder, more data, pretraining, more compute or ensembling | Ladder changes one thing at a time; ablations isolate loss, resolution and pretraining |
| P2. Weak baselines | Badly tuned U-Nets make new methods look good | Our U-Nets get the same loss, schedule and augmentation as every other model; nnU-Net 2D on the server as the strongest baseline |
| P3. Too few or unsuitable datasets | Median of three datasets; some datasets cannot separate methods | Five test sets, two of them from seen centres and three from unseen |
| P4. Inconsistent reporting | Small custom test sets, selective reporting, no cross-validation | Fixed saved splits, 3 seeds, confidence intervals, all test sets reported |

Selected results (Table 1, Dice %, VRAM in GB, training in GPU hours):

| Method | KiTS | AMOS | VRAM | GPU h |
|---|---|---|---|---|
| nnU-Net (original) | 86.04 | 88.64 | 7.7 | 9 |
| nnU-Net ResEnc M | 86.79 | 88.77 | 9.1 | 12 |
| nnU-Net ResEnc L | 88.17 | 89.41 | 22.7 | 35 |
| nnU-Net ResEnc XL | 88.67 | 89.68 | 36.6 | 66 |

Other findings from the same paper: the Transformer-based models tested (SwinUNETR, nnFormer, CoTr) do not reach even the original nnU-Net; the U-Mamba gains disappear when the Mamba layers are removed and the residual U-Net is kept. For us, the takeaway is the shape of the table: quality next to cost, so a reader can see what each step up the ladder costs.

## What we take, what we change

| Take | From | Use in |
|---|---|---|
| Padded convolutions with BatchNorm for our from-scratch U-Net, and a note on why we deviate from the 2015 paper | milesial | Stage 4 |
| Dice + BCE as the default loss; structure loss and plain BCE as ablations | nnU-Net, milesial, PraNet | Stages 4-7 |
| Split created once, saved to a file, reused by every run | nnU-Net | Stage 2 |
| Per-image metrics, with the reduction stated explicitly | SMP | Stage 2 |
| An explicit rule for empty prediction + empty target (Dice = 1) | milesial | Stage 2 |
| Quality-and-cost table, one change at a time | nnU-Net Revisited | Stage 7 |
| Pretrained-encoder U-Net as the second rung | SMP | Stage 5 |

What we do differently from milesial: evaluate every validation image (no `drop_last`) and compute metrics per image, not per batch.

Still open, decided in sub-step 1.4 (`decisions.md`): optimizer and schedule for our runs (AdamW with cosine decay vs SGD with poly), and how much of nnU-Net's augmentation is worth its cost at 256 px on the Air.

## Questions you should be able to answer

**Why do modern U-Nets use padded convolutions when the original did not?**
Unpadded convolutions shrink the output, so the original used overlapping tiles and cropped the skip features. Padding keeps input and output the same size, which removes the cropping and makes the code simpler. The cost is slightly less reliable predictions at image borders.

**What do skip connections do?**
The encoder loses spatial detail as it downsamples. Skip connections pass the high-resolution encoder features straight to the decoder, so it can place boundaries precisely while still using the deep, semantic features from the bottleneck.

**Why Dice + cross-entropy and not just one?**
Cross-entropy gives smooth, per-pixel gradients but is dominated by the many background pixels. Dice measures overlap directly and handles imbalance, but its gradients are unstable when the predicted mask is tiny. Together they train stably and still optimise what we measure.

**What is deep supervision?**
Extra losses on the decoder's lower-resolution outputs. They give the deeper layers a direct training signal. nnU-Net weights them 1, ½, ¼, … so the full-resolution output dominates.

**Why does nnU-Net beat newer architectures in "Revisited"?**
Because the comparison was fair: same data, same splits, a well-configured baseline. Many reported gains came from extra training budget, pretraining or a weak baseline, not from the new architecture.

**Pooled Dice or per-image Dice?**
Pooled Dice treats the test set as one big image, so large polyps dominate. Per-image Dice averages over images, so a small missed polyp costs as much as a large one. Clinically, missing a small polyp matters, so per-image is the better choice here. Report which one you use.

## Glossary

- **Encoder / decoder**: the downsampling half that extracts features, and the upsampling half that turns them back into a full-resolution mask.
- **Skip connection**: a direct link from an encoder level to the decoder level at the same resolution.
- **Transposed convolution**: a learned upsampling layer. The alternative is fixed interpolation (bilinear, nearest) followed by a normal convolution.
- **BatchNorm / InstanceNorm**: normalise activations using statistics over the batch, or per image. InstanceNorm does not depend on batch size, which matters for small batches.
- **Poly schedule**: learning rate decays as (1 − progress)^0.9, smoothly reaching zero at the end.
- **Foreground oversampling**: forcing a share of training patches to contain the object, so rare structures are seen often enough.
- **Cross-validation (k-fold)**: split training data into k parts, train k models each validated on a different part; every image is used for validation exactly once.
