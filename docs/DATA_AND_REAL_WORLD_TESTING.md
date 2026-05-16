# TinyLiveness Data And Real-World Testing

TinyLiveness should be trained and reported differently from TinyFaceMatch.
Face matching can be summarized with pair accuracy and verification thresholds;
liveness must show whether spoof attacks pass under real capture conditions.

## Dataset Plan

Start with this order:

1. **CelebA-Spoof** for large-scale RGB pretraining.
2. **MSU-MFSD** for mobile/laptop print and replay coverage.
3. **Replay-Attack** for a classic unseen cross-dataset test.
4. **OULU-NPU** or **SiW-Mv2** as a stricter real-world holdout.
5. A private consented device test set from the actual target phones/laptops.

Several datasets are not redistributable, and CelebA-Spoof is restricted to
non-commercial research/education. Do not publish trained weights unless the
full training-data license chain allows it.

## Convert Downloaded Data

The converter accepts raw folders or CSV manifests:

```bash
python training/prepare_liveness_dataset.py ^
  --source celeba_spoof=D:\datasets\CelebA-Spoof ^
  --source msu_mfsd=D:\datasets\MSU-MFSD ^
  --output-dir data/liveness ^
  --crop-face ^
  --video-stride 15 ^
  --max-frames-per-video 40 ^
  --overwrite
```

The output layout is:

```text
data/liveness/
  train/live
  train/spoof
  val/live
  val/spoof
  test/live
  test/spoof
  manifest.csv
```

The manifest keeps metadata columns such as `source`, `subject`, `attack_type`,
`device`, `lighting`, `environment`, and `frame_index`. If source folders do not
contain that metadata, the converter fills what it can infer from folder names.

## CelebA-Spoof Specific Flow

Download from the official Google Drive link after accepting the non-commercial
research/education terms:

```bash
python scripts/download_celeba_spoof.py --accept-noncommercial-terms
```

The official Google Drive package is a split zip archive. After all
`CelebA_Spoof.zip.001`, `.002`, ... parts are downloaded, combine and extract:

```bash
python scripts/extract_celeba_spoof.py
```

Then convert the extracted labels and images:

```bash
python training/prepare_celeba_spoof.py ^
  --source-dir data/raw/CelebA-Spoof/extracted ^
  --output-dir data/liveness_celeba_spoof ^
  --overwrite
```

For a quick pipeline test, limit each split:

```bash
python training/prepare_celeba_spoof.py ^
  --source-dir data/raw/CelebA-Spoof/extracted ^
  --output-dir data/liveness_celeba_spoof_small ^
  --max-per-label-per-split 5000 ^
  --overwrite
```

## Train

```bash
python training/train.py ^
  --data-dir data/liveness ^
  --train-split train ^
  --val-split val ^
  --epochs 30 ^
  --batch-size 64 ^
  --best-output checkpoints/best_tinyliveness.pt
```

## Real-World Evaluation

Use validation only to choose the threshold. Report results on a separate test
split, then break down errors by source and attack condition:

```bash
python training/evaluate_real_world.py ^
  --data-dir data/liveness ^
  --checkpoint checkpoints/best_tinyliveness.pt ^
  --calibration-split val ^
  --eval-split test ^
  --max-apcer 0.01 ^
  --group-by source,attack_type,device,lighting,environment
```

Outputs:

```text
reports/real_world_eval.csv
reports/real_world_eval.json
```

The report contains:

- ROC AUC
- accuracy and balanced accuracy
- APCER: spoof accepted as live
- BPCER: live rejected as spoof
- ACER: average of APCER and BPCER
- true live rate at APCER targets such as 1%, 0.5%, and 0.1%
- grouped results by source, attack type, device, lighting, and environment
- average inference latency per image

## What Counts As A Proper Real-Life Test

Use this pass/fail table before calling a checkpoint release-ready:

| Test | Requirement |
| --- | --- |
| Seen validation | Threshold selected on `val`, not `test` |
| Unseen source | At least one dataset source held out from training |
| Unseen attack | At least one attack type held out from training |
| Device shift | Test includes phones, laptops/webcams, and screens not used in training |
| Lighting shift | Test includes bright, dim, backlit, indoor, and outdoor samples |
| Real users | Test includes consented live captures from target devices |
| Security target | Report APCER at the chosen threshold, not only accuracy |
| Usability target | Report BPCER, because strict spoof blocking increases live-user retries |

Suggested first release target:

```text
APCER <= 1% on the real-world holdout
BPCER <= 10% on the real-world holdout
ACER <= 5%
CPU latency <= 30 ms per image on a mid-range laptop
```

These are engineering targets, not certification claims. Formal PAD conformance
requires lab testing such as ISO/IEC 30107-3.

## Private Capture Checklist

Capture only with consent. Keep raw images/videos private and out of logs.

Live samples:

- 50+ people minimum for a first internal test
- front camera and webcam captures
- glasses/no glasses, facial hair, masks/partial occlusion where appropriate
- bright, dim, backlit, indoor, and outdoor lighting

Spoof samples:

- printed photo on matte and glossy paper
- phone replay with low and high brightness
- laptop/monitor replay
- tablet replay
- cropped face photo and full-body photo
- angled, bent, and partially occluded presentation media

Keep at least one device and one spoof medium unseen until final test.
