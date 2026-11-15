# 1.1 The polyp segmentation benchmark: what the papers say and what the code does

Sub-step 1.1 of stage 1. Scope: the three sources that define the benchmark we will use (Kvasir-SEG, PraNet, Polyp-PVT). Every claim links to a paper section or to a line of code at a pinned commit.

Pinned commits:
- PraNet: [`95f12bd`](https://github.com/DengPingFan/PraNet/tree/95f12bd4e32ec954b5069654abae7de9473fd1f5)
- Polyp-PVT: [`cd71fbf`](https://github.com/DengPingFan/Polyp-PVT/tree/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4)

## Who is behind it

| Work | People and affiliation as printed in the paper | Source |
|---|---|---|
| Kvasir-SEG (MMM 2020) | Debesh Jha (SimulaMet; UiT The Arctic University of Norway), Pål Halvorsen (SimulaMet; OsloMet), Michael A. Riegler (SimulaMet), Thomas de Lange (University of Oslo; Oslo University Hospital), and others | [arXiv:1911.07069](https://arxiv.org/abs/1911.07069) |
| PraNet (MICCAI 2020, oral) | Deng-Ping Fan, Tao Zhou, Geng Chen, Huazhu Fu, Jianbing Shen (Inception Institute of AI, Abu Dhabi); Ge-Peng Ji (Wuhan University); Ling Shao | [arXiv:2006.11392](https://arxiv.org/abs/2006.11392) |
| Polyp-PVT (CAAI AIR 2023) | Bo Dong, Deng-Ping Fan (Nankai University); Wenhai Wang (Shanghai AI Laboratory); Jinpeng Li (IIAI); Huazhu Fu (A*STAR IHPC); Ling Shao (UCAS-Terminus AI Lab) | [arXiv:2108.06932](https://arxiv.org/abs/2108.06932) |

## The dataset

Kvasir-SEG: 1,000 colonoscopy images with polyp masks. An engineer and a medical doctor outlined each polyp in Labelbox, and an experienced gastroenterologist reviewed the outlines. Masks are 1-bit images, white foreground. (Kvasir-SEG paper, dataset section.)

The paper's own baseline (ResUNet, 80/10/10 random split, inputs resized to 320×320) reached Dice 0.788 and mean IoU 0.778 on its test split (Kvasir-SEG paper, Table 1).

## The benchmark as it exists in code

The papers describe a random 80/10/10 split (PraNet paper, Section 3.1). The released code does something else, and the code is what every later paper actually copied.

| | Images | Source |
|---|---|---|
| Train | 900 Kvasir-SEG + 550 CVC-ClinicDB = 1,450 | PraNet README, data section |
| Test, seen centres | Kvasir 100, CVC-ClinicDB 62 | same |
| Test, unseen centres | CVC-ColonDB 380, ETIS-LaribPolypDB 196, CVC-300 60 | same |
| Validation | none | see finding 1 |

The unseen-centre test sets are the interesting part: they measure whether a model generalises to other hospitals and other cameras.

### Training recipe (PraNet)

| Setting | Value | Code |
|---|---|---|
| Input size | 352×352, plus multi-scale training at 0.75, 1, 1.25 (rounded to multiples of 32) | [`MyTrain.py#L27-L40`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L27-L40), `trainsize` default at L80-81 |
| Loss | "structure loss" = boundary-weighted BCE + boundary-weighted IoU, applied to all 4 side outputs and summed | [`MyTrain.py#L12-L21`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L12-L21), L44-48 |
| Optimizer | Adam, lr 1e-4, batch 16, gradient clipping 0.5 | L76-83, L104 |
| Epochs | `range(1, opt.epoch)` with default 20, so 19 epochs are actually run | L74-75, L114 |
| Checkpoint | saved when `(epoch+1) % 10 == 0`, so epochs 9 and 19; the released file is `PraNet-19.pth` | L67-69 |
| Augmentation | none except multi-scale resizing | [`utils/dataloader.py#L19-L26`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/utils/dataloader.py#L19-L26) |
| Masks | resized with `transforms.Resize` (bilinear by default), so training targets are soft at edges | dataloader L24-26 |

The structure loss in one line: pixels near the boundary get more weight. The weight is `1 + 5 * |avgpool31(mask) - mask|`: 1 inside flat regions, about 3.5 at a straight edge (the 31×31 window is half polyp there), and up to 6 for thin or pointed parts (L13).

### Evaluation (PraNet)

1. The model output is upsampled to the original mask size and passed through sigmoid ([`MyTest.py#L35-L36`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTest.py#L35-L36)).
2. Each map is min-max stretched to [0, 1] per image (L37), then saved as a PNG.
3. A MATLAB script reads the PNGs and, for each image, computes Dice at 256 thresholds from 1 down to 0 ([`eval/main.m#L30`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/eval/main.m#L30), L100-107).
4. "meanDic" is the mean over images, then the mean over all 256 thresholds (L142-144). "maxDic" is the best threshold.

So the headline "mDice" is not Dice at threshold 0.5. It is Dice averaged over every threshold of a per-image stretched map.

## Published numbers

Mean Dice / mean IoU as printed. U-Net and U-Net++ rows are the numbers reported inside these papers, not by the U-Net authors.

| Method | Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 | Source |
|---|---|---|---|---|---|---|
| U-Net | 0.818 / 0.746 | 0.823 / 0.755 | 0.512 / 0.444 | 0.398 / 0.335 | 0.710 / 0.627 | PraNet Tables 1-2 |
| U-Net++ | 0.821 / 0.743 | 0.794 / 0.729 | 0.483 / 0.410 | 0.401 / 0.344 | 0.707 / 0.624 | PraNet Tables 1-2 |
| PraNet | 0.898 / 0.840 | 0.899 / 0.849 | 0.709 / 0.640 | 0.628 / 0.567 | 0.871 / 0.797 | PraNet Tables 1-2 |
| SANet | 0.904 / 0.847 | 0.916 / 0.859 | 0.753 / 0.670 | 0.750 / 0.654 | – | Polyp-PVT Tables IV-V |
| Polyp-PVT | 0.917 / 0.864 | 0.937 / 0.889 | 0.808 / 0.727 | 0.787 / 0.706 | – | Polyp-PVT Tables IV-V |

The pattern that matters for us: on seen centres the gap between U-Net and the best model is about 0.1 Dice; on unseen centres (ColonDB, ETIS) it is 0.3-0.4. Either generalisation is where architecture really helps, or the U-Net baseline was weak. Stage 4 and 5 test which.

## Findings

**1. No validation set, and Polyp-PVT selects its checkpoint on test data.**
PraNet simply keeps the last epoch ([`MyTrain.py#L67-L69`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L67-L69)). Polyp-PVT evaluates after every epoch on a folder called `test` inside `TestDataset` and keeps the best one ([`Train.py#L109-L121`](https://github.com/DengPingFan/Polyp-PVT/blob/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4/Train.py#L109-L121), default path L180-182). Choosing the best of 100 epochs on the test set makes the reported number optimistic. The Polyp-PVT paper does not describe a validation split either.

**2. "mDice" is not the Dice most people expect.**
See the evaluation section above. A model that is well calibrated at 0.5 and a model that is only good after stretching can get the same mDice. We will report both: Dice at 0.5 and the PraNet-style mDice.

**3. Paper and code disagree on the split.**
PraNet's paper says 80/10/10 random split; the released data uses fixed folders with 900/100 for Kvasir. Later papers follow the folders.

**4. Small inconsistencies across papers.**
PraNet on ColonDB is 0.709 in the PraNet paper (Table 2) and 0.712 in the Polyp-PVT paper (Table V). Polyp-PVT's Table VII lists different U-Net numbers on ColonDB (0.483) from its Table V (0.512). None of this changes conclusions, but it shows numbers get copied between papers rather than re-measured.
*Resolved in 3.2:* the released PraNet maps score 0.7115 on ColonDB with the authors' own evaluation, so 0.712 (Polyp-PVT) is the number the released maps give; the PraNet paper's 0.709 most likely came from an earlier set of maps ([report](../reports/stage3_published_maps.md)).

**5. Fragile but correct augmentation in Polyp-PVT.**
Image and mask get separate random transforms. They stay aligned only because the RNG is reseeded with the same seed before each ([`utils/dataloader.py#L59-L68`](https://github.com/DengPingFan/Polyp-PVT/blob/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4/utils/dataloader.py#L59-L68)). Augmentation is off by default (L24, `--augmentation` default `False`). A safer pattern is a single joint transform (albumentations does this).

**6. Latent learning-rate bug in PraNet.**
`adjust_lr` multiplies the current LR by `0.1 ** (epoch // decay_epoch)` every epoch ([`utils/utils.py#L20-L23`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/utils/utils.py#L20-L23)). Past `decay_epoch` that compounds every epoch instead of stepping once. With defaults (19 epochs, decay at 50) it never fires, so the LR is effectively constant.

**7. The CVC-ClinicDB test set shares videos with training (found in 2.2).**
CVC-ClinicDB has 612 frames from 29 videos. PraNet's split takes its 62 test frames from 26 of those videos and puts the remaining frames of the same videos in training. Test images are therefore near-copies of training images: their median similarity to the nearest training image is 0.942, against 0.845-0.861 for the unseen hospitals ([data inventory](../reports/data_inventory.md)). High CVC-ClinicDB scores (Polyp-PVT: 0.937) partly measure memory, not generalisation.

**8. The structure loss's BCE term is not boundary-weighted (found in 2.5).**
The loss computes BCE with `reduce='none'` ([`MyTrain.py#L14`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L14); Polyp-PVT copies it at [`Train.py#L17`](https://github.com/DengPingFan/Polyp-PVT/blob/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4/Train.py#L17)). `reduce` is a deprecated boolean argument, so PyTorch returns one averaged number, and multiplying it by the weight map and dividing by the map's sum gives back the same number. The "weighted BCE" is the plain mean BCE; only the IoU half of the loss uses the boundary weights. Our ablation runs both the published and the intended version ([decision D6](decisions.md)).

**9. The baselines' CVC-ColonDB scores leave out 21 images (found in 3.2).**
In the released maps for U-Net, U-Net++ and SFA, images 77-97 of CVC-ColonDB are saved as `.tif`, and the evaluation lists only `*.png` ([`eval/main.m#L52`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/eval/main.m#L52)). Those three methods were scored on 359 images; PraNet on all 380. Our scores reproduce the published ones only on the 359. With all 380 maps the baselines drop by 0.001 (U-Net++), 0.008 (U-Net) and 0.013 (SFA), so the table compares methods on different image sets, slightly in the baselines' favour ([report](../reports/stage3_published_maps.md)).

**10. The published tables reproduce from the released maps (3.2).**
Our port of the evaluation, run on the authors' maps, matches all 20 PraNet-paper numbers (4 methods, 5 test sets) within 0.0005 mDice and mIoU, except PraNet on ColonDB (finding 4). The ruler is the same; differences between our models and these numbers will be about the models.

**11. Kvasir does not reproduce exactly from the released weights (found in 3.3, open).**
Run on the Mac, the released PraNet weights reproduce the authors' maps within one 8-bit level on CVC-ClinicDB, CVC-ColonDB, ETIS and CVC-300, but on Kvasir every map differs a little and mDice is 0.894 instead of 0.898. The benchmark's Kvasir PNGs are pixel-identical to the original Kvasir-SEG JPEGs as Pillow decodes them, so our input is the published input. Kvasir is the only test set with JPEG originals, and the authors may have run on images decoded or prepared differently; a different decoder (Apple ImageIO) shifts single images by up to 0.17 mDice, which shows how sensitive the model is, but it does not move the mean towards the authors' number. The gap is within the 0.005 tolerance; it is reported, not hidden ([report](../reports/stage3_pranet_weights.md), [decoder check](../reports/stage3_kvasir_decoding.json)).

## What we take, what we change

| Take | Change |
|---|---|
| The five test sets and their seen/unseen split | Carve a fixed validation split from the 1,450 training images; select checkpoints on validation only |
| Structure loss, as one of the losses in our ablation | Report Dice at 0.5, IoU, Boundary IoU and PraNet-style mDice side by side |
| Released PraNet weights as a reference point | Re-run them through our own evaluation and check we reproduce Table 1-2 before trusting our code |
| 352 px as the comparison resolution | Also run 256 px, because the Air is memory-limited, and measure the cost of going lower |
| | 3 seeds, bootstrap confidence intervals, paired tests between models |
| | Joint image-mask augmentation; nearest-neighbour resizing for masks |

## Questions you should be able to answer

**Why not just use the numbers from the papers?**
Different papers compute Dice differently (threshold sweep vs fixed 0.5, per-image vs pooled) and some select checkpoints on the test set. Re-measuring everything with one evaluation script is the only fair comparison.

**What is the structure loss and why does it help?**
BCE and IoU with extra weight near object boundaries. Most pixels are easy background or easy interior; the errors that matter are at edges, so the loss concentrates there.

**Why do the unseen centres matter more than Kvasir?**
A hospital will deploy the model on its own equipment. ColonDB and ETIS come from different centres and cameras, so they approximate that situation. The gap between seen and unseen is the real measure of robustness.

**What is wrong with picking the best epoch on the test set?**
It turns the test set into a validation set. The number you report is then the maximum of many noisy measurements, which is biased upward. We will measure that bias directly in stage 7.

## Glossary

- **Dice**: 2·|A∩B| / (|A|+|B|). Overlap between predicted and true mask, 0 to 1.
- **IoU (Jaccard)**: |A∩B| / |A∪B|. Always ≤ Dice for the same masks.
- **mDice (PraNet style)**: Dice averaged over 256 thresholds after per-image min-max stretching, then averaged over images.
- **Seen / unseen centre**: whether images from that hospital were in the training data.
- **Side output / deep supervision**: extra predictions from intermediate decoder layers, each with its own loss.
