# Project 8: MVTec AD Anomaly Detection

Machine Learning Course — Computer Engineering Department, Sharif University  
**Lecturer:** Dr. Sharifi-Zarchi · **Spring 2025**  
**Author:** Amir Mohammad Fakhimi

---

## Overview

This project implements **anomaly detection** on the **MVTec AD** dataset: training only on normal (`train/good`) images and detecting anomalies at both **image** and **pixel** level. Two methods are implemented and compared:

1. **PaDiM** (baseline) — Patch Distribution Modeling with a Gaussian per spatial location  
2. **Student method** — a k-NN, PatchCore-style approach using the same ResNet backbone

---

## Dataset

- **MVTec AD** is used: 15 categories (e.g. bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper).
- For each category:
  - **Train:** only `train/good` (normal) images  
  - **Test:** `test/good` (normal) + `test/anomaly` (with defect types and pixel-level masks)

Download from: [MVTec AD Downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), then extract so that `mvtec_anomaly_detection/` contains the category folders.

---

## Implementation

### PaDiM (Baseline)

- **Backbone:** Pretrained ResNet18; feature maps from `layer1`, `layer2`, `layer3` are extracted and resized to a common spatial size (e.g. 28×28), then concatenated along channels.
- **Channel reduction:** A random subset of `d` channels (e.g. 100) is chosen once (with a fixed seed) to reduce dimension and match the paper.
- **Training (per category):** On train/good embeddings we compute:
  - **Mean** per spatial location: \( \mu \in \mathbb{R}^{C \times H \times W} \)
  - **Covariance** per location (with small diagonal regularization), then **inverse** for Mahalanobis distance.
- **Inference:** For each test image we get an embedding map, then at each pixel we compute the **Mahalanobis distance** to the learned \( (\mu, \Sigma) \) at that location. This gives a **pixel-level anomaly map**. The **image-level score** is obtained by aggregating the map (e.g. max or mean of top-k pixels).

So PaDiM models the distribution of normal patches with a Gaussian per location and uses distance to that distribution as the anomaly score.

### Student Method (k-NN / PatchCore-style)

- **Backbone:** Same as PaDiM (ResNet18, layers 1–3, aligned and concatenated, then channel subsampling with the same `d`).
- **Training (per category):**  
  - Extract patch features from all train/good images (each spatial location is one patch).  
  - Optionally reduce the set of patches with a **coreset** (greedy k-center) so that the memory bank stays manageable and diverse.  
  - Fit a **k-NN** model (Euclidean) on the (possibly coreset-reduced) patch set.
- **Inference:**  
  - For each test image, each patch is compared to the memory bank.  
  - Patch score = distance to the **k-th nearest neighbor** (PatchCore-style) or mean distance to k neighbors.  
  - These patch scores are reshaped to a spatial map → **pixel-level anomaly map**.  
  - **Image-level score** = aggregation (e.g. mean of top-k patch scores) then normalized to [0,1].

So the Student method treats anomaly detection as “how far is this patch from the normal patch set?” and uses the same backbone as PaDiM for a fair comparison.

---

## PaDiM vs Student — Comparison

| Aspect | PaDiM | Student |
|--------|--------|--------|
| **Model** | One Gaussian per spatial location (mean + covariance) | k-NN over a (possibly coreset) set of normal patches |
| **Score** | Mahalanobis distance to Gaussian | k-NN distance (e.g. k-th neighbor) in feature space |
| **Memory** | Fixed: one mean vector and one covariance per location | Grows with number of (coreset) patches |
| **Training** | One pass to compute mean/covariance | Extract patches → optional coreset → fit k-NN |
| **Inference** | Matrix ops per location | k-NN queries per patch |

In our runs, **PaDiM** achieves higher image and pixel metrics on average; the **Student** method is competitive and runs slightly faster, with the possibility to tune k, coreset size, and aggregation.

---

## Obtained Scores 

Scores are from the **macro mean across categories** (same as in the notebook leaderboard).

| Model   | ImageScore | PixelScore | OverallScore | runtime_sec (approx) |
|--------|------------|------------|--------------|----------------------|
| **PaDiM**  | 0.956      | 0.771      | **0.864**    | ~15.8 s              |
| **Student**| 0.920      | 0.717      | **0.818**    | ~13.6 s              |

- **ImageScore** = mean(Image AUROC, Image AP)  
- **PixelScore** = mean(Pixel AUROC, Pixel Best Dice)  
- **OverallScore** = mean(ImageScore, PixelScore)

So in this run, PaDiM leads on both image and pixel metrics and thus on OverallScore; the Student method is close and runs a bit faster. You can re-run the notebook and replace the table above with your latest numbers.

---


## Contributors

- **Alireza Sarbaz**
- **Nima Kahdenaroee**
- **Kian Amoo Khadem**
