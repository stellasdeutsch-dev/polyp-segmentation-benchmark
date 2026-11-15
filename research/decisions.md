# Decisions

The rules every later stage follows. Each decision says what we chose, why, where the idea comes from, and what we rejected. If a decision changes later, it is edited here with the reason, and the change gets its own commit.

Sources are the research notes: [1.1 polyp protocol](01_polyp_protocol.md), [1.2 U-Net lineage](02_unet_lineage.md), [1.3 foundation models](03_foundation_models.md).

Items marked **to confirm** depend on something we can only check once the data or environment exists. Each names the sub-step that checks it, and is marked **confirmed** once checked.

## Summary

| # | Topic | Decision |
|---|---|---|
| D1 | Data | PraNet benchmark folders: 1,450 training images, five fixed test sets |
| D2 | Validation | 158 images saved to a file: 90 Kvasir by image, 68 CVC-ClinicDB as whole video sequences that share no video with test (confirmed in 2.2) |
| D3 | Model selection | Best epoch on validation Dice at 0.5; test sets never used for selection |
| D4 | Metrics | Per-image Dice and IoU at 0.5, Boundary IoU, and PraNet-style mDice, computed at original resolution; tested against SMP, MONAI and the official Boundary IoU code (2.3) |
| D5 | Statistics | 3 seeds, bootstrap 95% CIs over test images, paired bootstrap between models |
| D6 | Loss | Dice + BCE (1:1); ablations: BCE only, structure loss |
| D7 | Optimizer | AdamW, cosine schedule with warmup, LR picked per model from three values on validation |
| D8 | Budget | 50 epochs for every model; about 32 min per run for rung 2 (2.5), 4-7 h for rung 1 at the paper's widths (4.2) |
| D9 | Resolution | 256 px main; 352 px for the best model; 448 px for the DINOv3 head |
| D10 | Augmentation | One light, joint image-mask augmentation for every model; ablation without it |
| D11 | Masks and normalisation | Binarise masks on load, nearest-neighbour resizing, ImageNet normalisation |
| D12 | Model ladder | U-Net from scratch (paper widths, matches milesial exactly, 4.1) → U-Net with ResNet-34 → linear head on frozen DINOv3 ViT-S; PraNet weights as reference; nnU-Net-width U-Net as a stage 7 ablation |
| D13 | Trust check | Our evaluation must reproduce PraNet's published numbers before our own results count |
| D14 | Reproducibility | Config, commit hash, versions and splits saved with every run; seeds plus CIs instead of bitwise repeats |
| D15 | Cost | Training time, peak memory (`torch.accelerator` counter, confirmed in 2.1), latency on MPS and CPU, parameter count |

## D1. Data

**Choice.** The PraNet benchmark as released: training on 900 Kvasir-SEG + 550 CVC-ClinicDB images; testing on Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS-LaribPolypDB (196), CVC-300 (60).

**Why.** It is what every polyp paper since 2020 reports on (1.1), so our numbers sit in one table with theirs. Three of the five test sets come from hospitals not seen in training, which is where models differ most (see the chart in [figures](figures/)).

**Rejected.** The random 80/10/10 split from the PraNet and Kvasir-SEG papers. The released code does not use it, and no later paper does either.

## D2. Validation split

**Choice (confirmed in 2.2).** 158 of the 1,450 training images, saved to `data/splits/val_seed0.json` and committed. Every run reads that file.
- **Kvasir:** 90 images (10%), sampled one by one with seed 0.
- **CVC-ClinicDB:** whole video sequences 11, 16 and 29 (68 frames, 12.4%). These are the only three sequences with no frame in the PraNet test set, so validation shares no video with either training or test.

Remaining training set: 1,292 images.

**Why.** PraNet has no validation set and Polyp-PVT picks its checkpoint on a test folder ([`Polyp-PVT Train.py#L109-L121`](https://github.com/DengPingFan/Polyp-PVT/blob/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4/Train.py#L109-L121)). A fixed, saved split is the nnU-Net habit ([`nnUNetTrainer.py#L668-L670`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L668-L670)).

**Correlated frames, and how 2.2 resolved them.** Frames of one CVC-ClinicDB video look alike, so a frame-level split would make validation too easy. The plan was to recover the clips from the images. That turned out not to be reliable on its own: frames inside a sequence are sparse samples, not continuous video, and only 15 of the 28 lowest-similarity frame pairs are true boundaries. The dataset itself publishes the answer: a table of frame ranges for its **29** sequences (Table 1 on the [CVC-ClinicDB page](https://polyp.grand-challenge.org/CVCClinicDB/); the PraNet paper's "31 clips" does not match it). We use that table (`seg.splits.CLINICDB_SEQUENCE_STARTS`) and check it against the images: ResNet-18 similarity of consecutive frames separates its 28 boundaries from within-sequence pairs with AUC 0.965.

**Finding: the CVC-ClinicDB test set is not a new-patient test.** Its 62 frames come from 26 of the 29 sequences, all of which also have training frames. Measured by the similarity of each image to its nearest training image, CVC-ClinicDB test sits far closer to training (median 0.942, 26% above 0.95) than Kvasir test (0.902) or the unseen hospitals (0.845-0.861). Results on it are reported, but labelled as near-duplicate performance. Details: [data inventory](../reports/data_inventory.md).

**Known limitation.** Kvasir-SEG has a few near-copies of its own: one validation image (`cju886ryxnsl50801r93jai7q`) is 0.999 similar to a training image. One image out of 90 does not change model selection; it stays in, and is listed in the report.

## D3. Model selection

**Choice.** Train for the full budget (D8) and keep the epoch with the best validation Dice at 0.5. Test sets are evaluated only once, on the selected checkpoint.

**One exception, for an experiment.** For the "selection optimism" experiment (stage 7), test Dice is also logged after every epoch, to a separate file that model selection never reads. Comparing "best epoch on test" with "epoch chosen on validation" measures how much test-set selection inflates a reported number.

## D4. Metrics

**Choice.**
- **Dice and IoU at threshold 0.5**, computed per image and averaged over images. These are the main numbers.
- **Boundary IoU** (Cheng, Girshick, Dollár, Berg, Kirillov, CVPR 2021, [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)): IoU computed only on mask pixels within distance d of each contour, with d = 2% of the image diagonal, the paper's setting (Sections 3 and 5). Mask IoU is dominated by interior pixels of large objects; Boundary IoU measures edge quality.
- **PraNet-style mDice**: a port of [`PraNet eval/main.m`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/eval/main.m#L30) (per-image min-max stretch, Dice at 256 thresholds, averaged). It is used only to compare with published numbers.
- **Resolution**: the prediction is upsampled to the original mask size before scoring, as PraNet does ([`MyTest.py#L35`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTest.py#L35)).
- **Empty prediction and empty mask**: Dice = IoU = 1, following milesial (1.2). Every test image contains a polyp, so this only matters for robustness.

**Why per image.** Pooled Dice is dominated by large polyps; per-image Dice gives a small missed polyp the same weight (1.2). It also matches how the papers average.

**Trust (done in 2.3).** Implemented in `seg.metrics`, 34 tests in `tests/test_metrics.py`:
- Dice and IoU equal SMP's `f1_score` / `iou_score` and MONAI's `compute_dice` / `compute_iou` on random masks of several sizes, including empty and full-frame masks. All three libraries also score an empty prediction on an empty mask as 1.
- Boundary regions are pixel-identical to the official Boundary IoU code ([`boundary_utils.py#L12-L30`](https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py#L12-L30), kept verbatim in `tests/reference/` under its BSD licence). It erodes d times with a 3×3 square; we do one (2d+1)×(2d+1) max-pool, which is equivalent and runs batched on the GPU. d is 7 pixels at 256 px and 10 at 352 px.
- PraNet-style mDice matches a slow line-by-line port of `MyTest.py`, `main.m` and `Fmeasure_calu.m`. Details that matter: the map is stretched, rounded to 8 bits (it is saved as a PNG) and stretched again; ground truth is foreground above 128; a pixel counts when its level is ≥ the threshold; Dice is 0 whenever there is no overlap.

**Details fixed in 2.3.**
- A pixel is foreground when probability ≥ 0.5 (inclusive, as in PraNet's `Fmeasure_calu.m`).
- PraNet thresholds are compared as exact 8-bit levels. Comparing them as floats instead drops pixels lying exactly on 66 of the 256 thresholds; on real mask shapes that changes per-image mDice by 0.0005 on average and at most 0.0016, well inside the 0.005 tolerance of D13. MATLAB's own rounding of `1:-1/255:0` cannot be checked here, so stage 3 is the real test.
  *Checked in 3.2:* `seg.metrics.matlab_colon` rebuilds the thresholds with MATLAB's two-ended colon rule (not run in MATLAB itself) (36 of 256 differ from k/255 in the last bits). Against exact 8-bit thresholds they move mDice by at most 0.00009 on the 20 published rows.
- Counting runs on the GPU; the division runs on the CPU in float64, because MPS has no float64.

**Found in 4.3: two places to threshold at 0.5.** For our models, Dice, IoU and Boundary IoU at 0.5 are taken on the raw probability, as in validation; `scripts/eval_run.py` reproduces the training loop's validation Dice to 6 decimals (0.564492 vs 0.564491 on a test checkpoint, CPU vs MPS), and a test pins the equality. PraNet's "Dice at 0.5" in 3.3 was taken on the stretched 8-bit map, because the saved map was all that was scored. Before PraNet appears next to our models in one table, its Dice at 0.5 is recomputed on its raw probability. mDice is unaffected: it is defined on the saved map for every model.

## D5. Statistics

**Choice.**
- Every configuration is trained with seeds 0, 1, 2. We report mean ± standard deviation across seeds.
- **95% bootstrap confidence interval** over test images: per-image Dice is averaged over the three seeds, then images are resampled with replacement 10,000 times.
- **Paired comparison** between two models: bootstrap on the per-image difference (same images, same resamples). A model counts as better only if the 95% interval of the difference excludes zero.
- Results are reported per test set, plus a pooled "unseen hospitals" figure over CVC-ColonDB, ETIS and CVC-300.

**Why.** Test sets are small (60 images for CVC-300). Without intervals, differences of 0.01 Dice look meaningful when they are noise. nnU-Net Revisited names inconsistent reporting as pitfall P4 (1.2).

## D6. Loss

**Choice.** Soft Dice (per image) + BCE, weight 1 each, for every model.
**Ablations (stage 7):** BCE only; PraNet's structure loss ([`MyTrain.py#L12-L21`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L12-L21)).

**Why.** Dice + cross-entropy is the nnU-Net default ([`nnUNetTrainer.py#L474-L482`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L474-L482)) and also what milesial uses. The structure loss is the polyp-specific alternative, so testing it answers whether that design choice matters on its own.

**Found in 2.5: the published structure loss does not weight its BCE term.** PraNet ([`MyTrain.py#L14`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L14)) and Polyp-PVT ([`Train.py#L17`](https://github.com/DengPingFan/Polyp-PVT/blob/cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4/Train.py#L17)) call `binary_cross_entropy_with_logits(..., reduce='none')`. `reduce` is a deprecated boolean argument; PyTorch reads the string as true, returns one averaged number, and the boundary weights then cancel out (checked on PyTorch 2.15: the "weighted" BCE equals the plain mean exactly). Only the IoU half is boundary-weighted. `seg.losses` therefore has two variants, `structure` (weights on both terms, as intended) and `structure_pranet` (as published, tested line for line against the original code). The stage 7 ablation runs both, which also answers whether the BCE weighting matters at all.

## D7. Optimizer and schedule

**Choice.**
- AdamW, weight decay 1e-4.
- Cosine decay to zero, with linear warmup over the first 5% of steps.
- Learning rate chosen per model from {1e-4, 3e-4, 1e-3} by validation Dice using seed 0, then kept for seeds 1 and 2.
- One learning rate for the whole model, including pretrained encoders (no layer-wise rates), to keep the comparison simple.

**Why.** nnU-Net's SGD + poly recipe is tuned for 1000 epochs × 250 iterations ([`nnUNetTrainer.py#L173-L181`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L173-L181)), far beyond a fanless laptop. PraNet and Polyp-PVT train with Adam / AdamW at 1e-4 for about 20-100 epochs (1.1). A small, fixed LR search per model is cheap and avoids the "weak baseline" pitfall (P2): every model gets its own tuned LR, by the same procedure.

**Rejected.** One shared LR for all models (unfair to whichever it suits least); a larger search (too costly on the Air).

## D8. Training budget

**Choice (confirmed in 2.5).** Every model gets **50 epochs** at the same resolution.

**Measured** (steady-state epochs, from [`reports/stage2_checks.json`](../reports/stage2_checks.json)). Rung 2 (SMP U-Net, ResNet-34, 24.4M parameters) at 256 px, batch 16: 37 s of training and 2 s of validation per epoch, peak memory 1.82 GB of the 11.84 GB macOS allows (both in units of 1,024³ bytes). So one 50-epoch run takes about 32 minutes. A bare training step runs at 36 images/s, so training is limited by GPU compute, not data loading.

**Why 50.** Between PraNet's ~20 epochs and Polyp-PVT's 100 (1.1), and it keeps a run inside a single session on the Air (runs can also stop with `max_minutes` and resume). **Check before any comparison:** if validation Dice is still rising over the last 10 epochs of the first full run (stage 5), the budget is raised for every model. Rung 1's epoch time is measured in stage 4 before its runs start.

**Rung 1 measured (4.2).** The from-scratch U-Net at the paper's widths takes 302-491 s of training per epoch (three epochs; the spread is most likely heat on the fanless laptop), about 4-7 hours per 50-epoch run and 20-35 hours for the five runs D7 asks for. A step benchmark ([`reports/stage4_timing.json`](../reports/stage4_timing.json), `make unet-timing`) puts it at 3.9× the ResNet-34 U-Net's time per image (96.3 vs 15.6 GFLOP per forward pass). float16 or bfloat16 autocast saves under 6%, and channels-last memory makes MPS 2.3× slower, so neither is used. nnU-Net's 2D widths, 32 to 512 channels ([`default_experiment_planner.py#L50`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py#L50), [`#L64`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py#L64)), would be 2.3× faster. Decision: rung 1 keeps the paper's widths and the full 50-epoch budget, so nobody can say the baseline was shrunk; the 32-512 U-Net is added to stage 7 as an ablation of width. Runs go through `scripts/run_queue.py` with a 10-minute rest between runs.

**Two slowdowns found while timing, both fixed.** The first timing run took 65 s + 25 s per epoch. (1) Loader workers were restarted every epoch; on macOS each new worker re-imports PyTorch and albumentations. Workers now persist across epochs. (2) `loss.item()` in every step made the CPU wait for the GPU; the loss is now summed on the GPU and read once per epoch. Together: 2.3× faster, and the first epoch's loss is identical before and after, so the fix changed speed, not what is computed.

## D9. Resolution

**Choice.**
- 256 px for all main runs (memory and time on the Air).
- 352 px, PraNet's size ([`MyTrain.py#L80-L81`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L80-L81)), for the best model only, to measure what we lose at 256.
- 448 px for the DINOv3 head as well, because its 16-pixel patches give only a 16×16 grid at 256 px (1.3).

## D10. Augmentation

**Choice.** One joint image-and-mask transform (albumentations), the same for every model:
- horizontal and vertical flip, p = 0.5 each;
- affine: scale 0.75-1.25, rotation ±15°, p = 0.5;
- brightness and contrast ±20%, p = 0.5.

**Ablation (stage 7):** no augmentation.

**Why.** The scale range follows PraNet's multi-scale training (0.75, 1, 1.25; [`MyTrain.py#L27`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py#L27)) and sits inside nnU-Net's 0.7-1.4 ([`nnUNetTrainer.py#L820-L821`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L820-L821)). The brightness and contrast range matches nnU-Net's 0.75-1.25 ([`#L848-L855`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L848-L855)). Elastic deformation is left out, as nnU-Net also does by default ([`#L819`](https://github.com/MIC-DKFZ/nnUNet/blob/90cc65aa1958bd13678cc8c7b3b9496ee6f07d7e/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L819)). A joint transform avoids the reseeding trick Polyp-PVT relies on to keep image and mask aligned (1.1).

## D11. Masks and normalisation

**Choice.**
- Masks are binarised on load (> 127 → 1) and resized with nearest-neighbour interpolation, so training targets stay binary.
- Images are normalised with ImageNet mean and standard deviation for every model. That is what PraNet uses ([`dataloader.py#L22-L23`](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/utils/dataloader.py#L22-L23)) and what DINOv3 expects ([foreground notebook](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/foreground_segmentation.ipynb)).

**Rejected.** PraNet's bilinear mask resizing, which produces soft edge labels. Binary targets keep the loss and the metrics consistent.

## D12. Model ladder

| Rung | Model | Key settings | Stage |
|---|---|---|---|
| 1 | U-Net from scratch, our own code | milesial-style design: padded 3×3 convolutions, BatchNorm, transposed-conv upsampling, 64 → 1,024 channels; He initialisation as in the 2015 paper | 4 |
| 2 | SMP `Unet` with ResNet-34 | ImageNet weights, library defaults (decoder 256-16 channels, nearest upsampling) | 5 |
| 3 | Linear head on frozen DINOv3 ViT-S/16 | last-layer features, BatchNorm + 1×1 conv + bilinear upsampling (official head); features cached once | 6 |
| Ref | PraNet, released weights | evaluated only, never trained | 3 |

**Rung 1 built (4.1).** `seg.unet.UNet`, 31,037,633 parameters. With milesial's weights copied in, it gives bit-identical outputs in eval and train mode and the same gradients (`tests/test_unet.py`), so it is the same function as the reference, written independently.

**Licence fallback.** If DINOv3 access is not granted, rung 3 uses DINOv2 ViT-S/14, labelled in every table. Its 14-pixel patches give a slightly finer grid at the same input size.

## D13. Trust check before any result counts

**Choice.** Run the released PraNet weights through our evaluation on all five test sets. Our PraNet-style mDice must match PraNet's Tables 1-2 within 0.005 on every set.

**Why.** If our evaluation cannot reproduce a published number from published weights, any difference between our models and the literature could be a bug in our evaluation.

**Result, part 1 (3.2): the evaluation code.** Before running any weights, we scored the prediction maps the authors released with the paper. Same files, so only the evaluation code differs. All 20 pairs (PraNet, U-Net, U-Net++, SFA on five test sets) pass: 19 within 0.0005, the largest gap 0.0025 (PraNet on CVC-ColonDB, where the paper's 0.709 most likely predates the released maps; Polyp-PVT reports 0.712, we get 0.7115). mIoU agrees within 0.0005 everywhere. Two details mattered: main.m reads only `*.png` (21 baseline maps are `.tif`, research finding 9), and it upsamples wrong-size maps bicubically (all 196 SFA maps on ETIS). Part 2 (3.3) runs the released weights.

**Result, part 2 (3.3): the whole pipeline.** The released weights, run on the Mac by our code following MyTest.py, give mDice within 0.005 of the paper on all five test sets (largest gaps: -0.0036 on Kvasir, +0.0025 on CVC-ColonDB, the second inherited from 3.2). On the four sets with lossless originals, our maps equal the authors' within one 8-bit level on all 698 images: preprocessing, inference on Apple's GPU, saving and scoring all do what the authors' code did. The Apple GPU and the CPU agree to 0.00001 per image. Open: on Kvasir, the only set that started as JPEG, every map differs slightly from the authors' and mDice is 0.0038 lower; see research finding 11. D13 is met, so stage 4 can start.

## D14. Reproducibility

**Choice.**
- Seeds set for Python, NumPy and PyTorch.
- Each run saves its config, git commit hash, package versions and split files next to its checkpoint.
- Splits are committed to the repository; data and checkpoints are not.
- Every table and figure is produced by a script from saved result files, never typed by hand.

**Honest limit.** GPU training on Apple's MPS backend is not guaranteed to be bit-for-bit repeatable. We rely on three seeds and confidence intervals, not on identical reruns.

**Found in 2.5: augmentation seeding.** albumentations 2.x draws from its own generator, which `np.random.seed` does not reach; and with several loader workers, every worker started from the same seed and repeated the others' draws (with 2 workers, only 4 of 8 augmentations were distinct). `seg.data.make_loader` now gives each worker its own seed derived from PyTorch's per-worker seed: draws are distinct and a rerun with the same seed repeats them exactly (tested with 0 and 2 workers in `tests/test_data.py`).

**Settings are type-checked.** PyYAML reads `3e-4` (no decimal point) as text, so `--set lr=3e-4` would have reached the optimizer as a string. `TrainConfig` converts every value to its declared type and rejects anything it cannot read.

**Found in the 4.5 check: resuming.** A resumed run used to reseed its training loader with the run's seed and so replayed the shuffles and augmentations of its first epochs. `seg.train.loader_seed` now gives a resumed run its own seed (run seed and resume epoch), and `summary.json` lists the epochs after which a run was resumed. A resumed run is still not bit-identical to an uninterrupted one; reports flag resumed runs.

## D15. Cost measurement

**Choice.** For every model:
- wall-clock training time;
- peak memory during training;
- inference latency: median over 100 images after 10 warm-up runs, batch 1, at 256 px, on MPS and on CPU, measured after the machine has cooled down;
- parameter count, total and trainable.

**Confirmed in 2.1.** `torch.mps` has no peak counter in PyTorch 2.15, but the device-agnostic `torch.accelerator.max_memory_allocated()` works on MPS. After `reset_peak_memory_stats()`, allocating 256 MB and then 512 MB read 256 and 768 MB exactly; the peak held at 768 MB after freeing; a convolution forward and backward pass peaked at 897 MB while holding 130 MB afterwards (about the 896 MB of saved activations expected). `torch.mps.driver_allocated_memory()` grows in 1 GB steps and is too coarse to report. Implemented as `seg.env.measure`, tested in `tests/test_env.py`. macOS recommends at most 11.84 GB of GPU memory on this machine, which bounds our batch sizes.

**Found in 4.3.** Other work on the laptop slows training badly: tests and smoke tests during one U-Net epoch made it 6.4× slower (1,890 s against 294 s). Training cost is therefore reported as the median epoch time times the number of epochs, next to the wall-clock total, and nothing heavy runs alongside training.

**Why.** nnU-Net Revisited reports quality next to VRAM and GPU hours (1.2), and EoMT reports FPS next to mIoU (1.3). A model that is 0.01 Dice better but ten times slower is a different trade-off, and the table should show it.
